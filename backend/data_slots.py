"""
FinBrain 高级数据 API 插槽系统

提供多数据源熔断回退机制，类似 LLM Slot 1/2/3。
支持的 provider（逐步完善）：
    - tushare: Tushare Pro（需 pip install tushare + token）
    - wind: Wind 终端 API（需本地安装 Wind 并授权）
    - choice: 东方财富 Choice 终端 API
    - ifind: 同花顺 iFinD API
    - bloomberg: Bloomberg API / BPIPE（机构级）

用法：
    from backend.data_slots import query_data_slot
    fin = query_data_slot("financials", "300502")
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional


# ============================================================
#  配置读取
# ============================================================

def _read_data_slots() -> list[dict]:
    """从环境变量读取 3 个数据插槽配置。slot 1 可选，不强制。"""
    slots = []
    for i in range(1, 4):
        prefix = f"DATA_SLOT_{i}"
        provider = os.getenv(f"{prefix}_PROVIDER", "").strip().lower()
        api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
        base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
        extra = os.getenv(f"{prefix}_EXTRA", "").strip()
        if not provider:
            continue
        slots.append({
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "extra": extra,
        })
    return slots


# ============================================================
#  抽象基类
# ============================================================

class BaseDataProvider(ABC):
    """高级数据源统一接口。"""

    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.extra = config.get("extra", "")
        self._client = None

    @property
    @abstractmethod
    def name(self) -> str:
        """provider 英文标识"""
        ...

    @property
    def connected(self) -> bool:
        """是否已连接/可用。子类可覆盖做真实探测。"""
        return bool(self.api_key)

    def _error(self, msg: str, detail: Any = None) -> dict:
        return {"error": msg, "provider": self.name, "detail": detail}

    # ---- 核心数据接口 ----
    def fetch_stock_price(self, symbol: str) -> dict:
        return self._error(f"{self.name} 未实现 fetch_stock_price")

    def fetch_financials(self, symbol: str) -> dict:
        return self._error(f"{self.name} 未实现 fetch_financials")

    def fetch_valuation(self, symbol: str) -> dict:
        return self._error(f"{self.name} 未实现 fetch_valuation")

    def fetch_industry_info(self, symbol: str) -> dict:
        return self._error(f"{self.name} 未实现 fetch_industry_info")

    def fetch_fund_flow(self, symbol: str) -> dict:
        return self._error(f"{self.name} 未实现 fetch_fund_flow")

    # ---- 深度/高级数据接口 ----
    def fetch_management(self, symbol: str) -> Optional[dict]:
        return None

    def fetch_institutional_holdings(self, symbol: str) -> Optional[dict]:
        return None

    def fetch_supply_chain(self, symbol: str) -> Optional[dict]:
        return None

    def fetch_esg(self, symbol: str) -> Optional[dict]:
        return None

    def fetch_alternative(self, symbol: str) -> Optional[dict]:
        return None


# ============================================================
#  Tushare Pro 适配器
# ============================================================

class TushareProvider(BaseDataProvider):
    """Tushare Pro 适配器。需要安装 tushare 并配置有效 token。"""

    @property
    def name(self) -> str:
        return "tushare"

    @property
    def connected(self) -> bool:
        try:
            return self._get_client() is not None
        except Exception:
            return False

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import tushare as ts
        except ImportError as e:
            raise RuntimeError("tushare 未安装，请执行 pip install tushare") from e
        if not self.api_key:
            raise RuntimeError("Tushare token 未配置")
        self._client = ts.pro_api(self.api_key)
        return self._client

    def _ts_code(self, symbol: str) -> str:
        """A股代码 → Tushare ts_code"""
        if symbol.startswith(("60", "68")):
            return f"{symbol}.SH"
        elif symbol.startswith(("00", "30", "32", "20")):
            return f"{symbol}.SZ"
        elif symbol.startswith(("8", "4", "43")):
            return f"{symbol}.BJ"
        return f"{symbol}.SH"

    def _call(self, api_name: str, **kwargs):
        """封装 Tushare 调用，异常时返回 {"error": ...}。"""
        try:
            pro = self._get_client()
            df = getattr(pro, api_name)(**kwargs)
            if df is None or df.empty:
                return {"error": f"{api_name} 返回空数据", "provider": self.name}
            return df
        except Exception as e:
            return {"error": str(e), "provider": self.name}

    def fetch_stock_price(self, symbol: str) -> dict:
        ts_code = self._ts_code(symbol)
        today = self._call("trade_cal", exchange="SSE", start_date="20260101", end_date="20261231", is_open="1")
        if isinstance(today, dict) and "error" in today:
            return today
        trade_dates = today["cal_date"].tolist() if hasattr(today, "__getitem__") and "cal_date" in today else []
        if not trade_dates:
            return self._error("无法获取最近交易日")
        latest_date = trade_dates[-1]
        df = self._call("daily", ts_code=ts_code, trade_date=latest_date)
        if isinstance(df, dict) and "error" in df:
            return df
        row = df.iloc[0]
        return {
            "name": symbol,
            "open": float(row.get("open", 0)),
            "yesterday_close": float(row.get("pre_close", 0)),
            "price": float(row.get("close", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "time": str(row.get("trade_date", latest_date)),
            "provider": self.name,
        }

    def fetch_financials(self, symbol: str) -> dict:
        ts_code = self._ts_code(symbol)
        end_date = "20261231"
        start_date = "20240101"

        inc = self._call("income", ts_code=ts_code, start_date=start_date, end_date=end_date)
        bal = self._call("balancesheet", ts_code=ts_code, start_date=start_date, end_date=end_date)
        cf = self._call("cashflow", ts_code=ts_code, start_date=start_date, end_date=end_date)

        if isinstance(inc, dict) and "error" in inc:
            return inc
        if isinstance(bal, dict) and "error" in bal:
            return bal
        if isinstance(cf, dict) and "error" in cf:
            return cf

        # 列名映射到 tools.py 内部格式
        INC_MAP = {
            "total_revenue": "营业总收入",
            "oper_cost": "营业成本",
            "n_income_attr_p": "归母净利润",
            "profit_dedt": "扣非净利润",
            "sell_exp": "销售费用",
            "admin_exp": "管理费用",
            "fin_exp": "财务费用",
            "operate_profit": "营业利润",
            "income_tax": "所得税",
        }
        BAL_MAP = {
            "total_assets": "资产总计",
            "total_liab": "负债合计",
            "total_hldr_eqy_exc_min_int": "股东权益",
            "fix_assets": "固定资产",
            "money_cap": "货币资金",
            "accounts_receiv": "应收账款",
            "inventories": "存货",
            "acct_payable": "应付账款",
        }
        CF_MAP = {
            "n_cashflow_act": "经营现金流净额",
            "n_cashflow_inv_act": "投资现金流净额",
            "n_cashflow_fin_act": "筹资现金流净额",
            "c_paid_for_invest": "购建固定资产支付现金",
            "depr_fa_coga_dpba": "折旧摊销",
        }

        def _transform(df, col_map):
            if df is None or df.empty:
                return []
            try:
                from pandas import notnull as pd_notnull
            except ImportError:
                pd_notnull = lambda x: x is not None
            records = []
            for _, row in df.iterrows():
                record = {"报告期": str(row.get("end_date", ""))}
                for eng, cn in col_map.items():
                    if eng in row:
                        val = row[eng]
                        record[cn] = float(val) if pd_notnull(val) else None
                # 资产负债率计算
                if "资产总计" in record and "负债合计" in record:
                    assets = record.get("资产总计")
                    liab = record.get("负债合计")
                    if assets and liab:
                        record["资产负债率"] = round(liab / assets * 100, 2)
                records.append(record)
            return records

        return {
            "balance": _transform(bal, BAL_MAP),
            "profit": _transform(inc, INC_MAP),
            "cashflow": _transform(cf, CF_MAP),
            "provider": self.name,
        }

    def fetch_valuation(self, symbol: str) -> dict:
        ts_code = self._ts_code(symbol)
        df = self._call("daily_basic", ts_code=ts_code)
        if isinstance(df, dict) and "error" in df:
            return df
        row = df.iloc[0]
        return {
            "symbol": symbol,
            "data": [{
                "日期": str(row.get("trade_date", "")),
                "PE(TTM)": float(row.get("pe_ttm", 0)) if row.get("pe_ttm") is not None else None,
                "PB": float(row.get("pb", 0)) if row.get("pb") is not None else None,
                "总市值(亿)": float(row.get("total_mv", 0)) / 1e4 if row.get("total_mv") is not None else None,
                "ROE(%)": float(row.get("roe", 0)) if row.get("roe") is not None else None,
                "每股收益": float(row.get("eps", 0)) if row.get("eps") is not None else None,
            }],
            "provider": self.name,
        }

    def fetch_industry_info(self, symbol: str) -> dict:
        ts_code = self._ts_code(symbol)
        df = self._call("stock_company", ts_code=ts_code)
        if isinstance(df, dict) and "error" in df:
            return df
        row = df.iloc[0]
        return {
            "symbol": symbol,
            "name": row.get("name", ""),
            "industry_name": row.get("industry", ""),
            "industry_code": "",
            "index_trend": [],
            "provider": self.name,
        }

    def fetch_fund_flow(self, symbol: str) -> dict:
        # Tushare 缺少免费的个股实时资金流向接口，返回明确说明
        return self._error("Tushare Pro 免费版不支撑实时个股资金流向，建议用 ths 免费源或 Choice 终端")


# ============================================================
#  其他主流高级数据源桩（用户接入自有授权后补全）
# ============================================================

class WindProvider(BaseDataProvider):
    @property
    def name(self) -> str:
        return "wind"

    @property
    def connected(self) -> bool:
        try:
            import WindPy
            return WindPy.w.isconnected()
        except Exception:
            return False

    def _error(self, msg: str, detail: Any = None) -> dict:
        return {"error": msg, "provider": "wind", "detail": detail,
                "doc": "https://www.wind.com.cn/"}


class ChoiceProvider(BaseDataProvider):
    @property
    def name(self) -> str:
        return "choice"

    def _error(self, msg: str, detail: Any = None) -> dict:
        return {"error": msg, "provider": "choice", "detail": detail,
                "doc": "https://choice.eastmoney.com/"}


class iFinDProvider(BaseDataProvider):
    @property
    def name(self) -> str:
        return "ifind"

    def _error(self, msg: str, detail: Any = None) -> dict:
        return {"error": msg, "provider": "ifind", "detail": detail,
                "doc": "https://www.51ifind.com/"}


class BloombergProvider(BaseDataProvider):
    @property
    def name(self) -> str:
        return "bloomberg"

    def _error(self, msg: str, detail: Any = None) -> dict:
        return {"error": msg, "provider": "bloomberg", "detail": detail,
                "doc": "https://www.bloomberg.com/professional/"}


# ============================================================
#  Provider 工厂
# ============================================================

_PROVIDER_MAP = {
    "tushare": TushareProvider,
    "wind": WindProvider,
    "choice": ChoiceProvider,
    "ifind": iFinDProvider,
    "bloomberg": BloombergProvider,
}


def _provider_for(config: dict) -> BaseDataProvider:
    provider = config.get("provider", "").lower()
    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        raise ValueError(f"未知的数据 provider: {provider}")
    return cls(config)


# ============================================================
#  熔断回退链
# ============================================================

class DataAPIFallbackChain:
    """按槽位顺序尝试高级数据源，任一成功即返回。"""

    def __init__(self, slots: list[dict]):
        self.slots = slots
        self._providers = [_provider_for(s) for s in slots]

    def invoke(self, category: str, symbol: str) -> dict:
        last_err = None
        trace = []
        for provider in self._providers:
            try:
                method_name = f"fetch_{category}"
                method = getattr(provider, method_name, None)
                if method is None:
                    raise AttributeError(f"{provider.name} 未实现 {method_name}")
                result = method(symbol)
                if isinstance(result, dict) and "error" not in result:
                    result["_data_slot_trace"] = trace + [{"provider": provider.name, "status": "success"}]
                    return result
                err = result.get("error", "unknown") if isinstance(result, dict) else str(result)
                trace.append({"provider": provider.name, "status": "failed", "error": err})
                last_err = err
            except Exception as e:
                trace.append({"provider": provider.name, "status": "failed", "error": str(e)})
                last_err = str(e)
        return {"error": f"全部 {len(self._providers)} 个数据插槽失败。最后错误: {last_err}",
                "_data_slot_trace": trace}


# ============================================================
#  统一入口 + 单例
# ============================================================

_CHAIN = None


def _get_chain() -> Optional[DataAPIFallbackChain]:
    global _CHAIN
    if _CHAIN is None:
        slots = _read_data_slots()
        if slots:
            _CHAIN = DataAPIFallbackChain(slots)
    return _CHAIN


def query_data_slot(category: str, symbol: str) -> dict:
    """
    统一的高级数据插槽查询入口。
    category: stock_price | financials | valuation | industry_info | fund_flow |
              management | institutional_holdings | supply_chain | esg | alternative
    """
    chain = _get_chain()
    if chain is None:
        return {"error": "未配置任何数据插槽 (DATA_SLOT_*)", "_data_slot_trace": []}
    return chain.invoke(category, symbol)


def list_configured_slots() -> list[dict]:
    """返回当前已配置的数据插槽信息（不含 API Key）。"""
    slots = _read_data_slots()
    return [{"provider": s["provider"], "has_key": bool(s["api_key"]),
             "base_url": s["base_url"], "extra": s["extra"]} for s in slots]


# 高级数据类别名称（供 datasource_tier 使用）
CATEGORY_MAP = {
    "管理层画像": "management",
    "机构持仓": "institutional_holdings",
    "产业链图谱": "supply_chain",
    "ESG与治理": "esg",
    "另类数据": "alternative",
}
