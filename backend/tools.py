"""
FinBrain 数据工具层 — 纯函数，不依赖任何 Agent 框架。
可直接被 MCP Server、LangGraph Agent、或命令行脚本调用。
"""

import os
import time
import urllib.request
import urllib.parse
import ssl
import re
import json
import logging
import pandas as pd
import akshare as ak

# FIX-01/02 股本SSOT：TTM EPS 统一入口；FIX-06 涨跌停规则引擎
from backend.share_registry import compute_ttm_eps
from backend.trading_rules import is_limit_up as _rules_is_limit_up

logger = logging.getLogger(__name__)

# ============================================================
#  数据源配置（从 .env 注入，默认免费源）
# ============================================================

_SOURCE_DEFAULTS = {
    "stock_price":   "sina",           # 新浪实时行情
    "financials":    "eastmoney",      # 东方财富 datacenter
    "industry":      "eastmoney_ths",  # 东方财富+同花顺
    "fund_flow":     "ths",            # 同花顺 10jqka
}

_SUPPORTED_SOURCES = {
    "stock_price":   {"sina": "新浪财经", "akshare": "AkShare"},
    "financials":    {"eastmoney": "东方财富 datacenter", "akshare": "AkShare"},
    "industry":      {"eastmoney_ths": "东方财富+同花顺", "akshare": "AkShare"},
    "fund_flow":     {"ths": "同花顺 10jqka", "akshare": "AkShare"},
}


# ---- 数据模式：local(本地直连) / remote(调API) ----
_DATA_MODE = os.getenv("FINBRAIN_DATA_MODE", "local")
_DATA_API = os.getenv("FINBRAIN_DATA_API", "http://localhost:8000")

# 远程模式端点映射：函数名 → API 路径
# Harness: 工具调用追踪（去重）
_called_tools: dict[str, dict] = {}  # {symbol: {tool_name: {ts, result}}}
import threading as _threading
_call_lock = _threading.Lock()

# 去重 TTL 与缓存一致（实时数据短，财报长）
_DEDUP_TTL = {
    "stock_price": 30, "stock_history": 300, "fund_flow": 300,
    "intraday": 30, "financial_statements": 1800, "valuation": 1800,
    "industry_info": 3600, "screen_stocks": 600, "limit_up_pool": 60,
    "concept_ranking": 600, "dragon_tiger_list": 300, "dragon_tiger_detail": 300,
}

def _dedup_check(symbol: str, tool_name: str) -> dict | None:
    """TTL 感知去重：在缓存有效期内不重复调用。实时数据(30s)可重调，财报(30min)不可。"""
    with _call_lock:
        if symbol in _called_tools and tool_name in _called_tools[symbol]:
            entry = _called_tools[symbol][tool_name]
            ttl = _DEDUP_TTL.get(tool_name, 60)
            if time.time() - entry["ts"] < ttl:
                return {"_dedup": True, "_cached": entry["result"],
                        "info": f"{tool_name}({symbol}) TTL内({ttl}s)，跳过"}
    return None

def _dedup_record(symbol: str, tool_name: str, result: dict):
    """记录工具调用结果（含时间戳）"""
    with _call_lock:
        if symbol not in _called_tools:
            _called_tools[symbol] = {}
        _called_tools[symbol][tool_name] = {"ts": time.time(), "result": result}

def _clear_dedup():
    """清空去重记录（每次新分析开始时调用）"""
    with _call_lock:
        _called_tools.clear()
        _fail_counts.clear()


# Harness: 失败计数器——同一工具连续失败3次→熔断
_fail_counts: dict[str, int] = {}

def _should_skip(tool_name: str) -> bool:
    """连续失败3次→返回True，该工具本轮不再调用"""
    return _fail_counts.get(tool_name, 0) >= 3

def _record_failure(tool_name: str):
    _fail_counts[tool_name] = _fail_counts.get(tool_name, 0) + 1

def _record_success(tool_name: str):
    if tool_name in _fail_counts:
        _fail_counts[tool_name] = 0


# ---- 数据插槽回退：免费源失败时尝试高级数据源 ----
def _is_result_ok(result: dict, required_keys: list[str] | None = None) -> bool:
    """判断一个数据结果是否可用。"""
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    if required_keys:
        for k in required_keys:
            if k not in result or result[k] in (None, "", []):
                return False
    return True


def _try_with_data_slots(symbol: str, primary_result: dict, category: str,
                         required_keys: list[str] | None = None) -> dict:
    """免费源结果失败/缺失时，尝试 DATA_SLOT_* 高级数据源回退。"""
    if _is_result_ok(primary_result, required_keys):
        return primary_result
    if _DATA_MODE == "remote":
        # remote 模式下由服务端统一处理，客户端不额外调用
        return primary_result
    try:
        from backend.data_slots import query_data_slot
        fallback = query_data_slot(category, symbol)
        if _is_result_ok(fallback, required_keys):
            fallback["_fallback_from"] = "data_slots"
            return fallback
    except Exception:
        pass
    return primary_result


_REMOTE_ROUTES = {
    "fetch_stock_price":      ("api/data/stock_price",      ["symbol"]),
    "fetch_stock_history":    ("api/data/stock_history",    ["symbol", "scale", "datalen"]),
    "get_financial_statements":("api/data/financials",      ["symbol"]),
    "get_valuation":          ("api/data/valuation",         ["symbol"]),
    "get_industry_info":      ("api/data/industry",          ["symbol"]),
    "get_fund_flow":          ("api/data/fund_flow",         ["symbol"]),
    "get_limit_up_pool":      ("api/market/limit_up",        ["top_n"]),
    "get_market_breadth":     ("api/market/breadth",         []),
    "get_sector_fund_flow":   ("api/market/sector_fund_flow",["top_n"]),
    "get_intraday":           ("api/data/intraday",           ["symbol"]),
    "get_concept_ranking":    ("api/market/concept_ranking",  ["top_n"]),
    "get_dragon_tiger_list":  ("api/market/dragon_tiger_list",["date"]),
    "get_dragon_tiger_detail":("api/market/dragon_tiger_detail",["symbol"]),
    "get_stock_streak":       ("api/data/stock_streak",       ["symbol"]),
    "screen_stocks":          ("api/market/screen_stocks",    ["max_pe","max_pb","min_mktcap","top_n"]),
}


def _remote_fetch(endpoint: str, params: dict = None) -> dict:
    """远程模式：调 Data API 获取数据"""
    import urllib.request as _ur
    url = f"{_DATA_API.rstrip('/')}/{endpoint.lstrip('/')}"
    if params:
        import urllib.parse as _up
        qs = _up.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{url}?{qs}"
    try:
        req = _ur.Request(url, headers={"User-Agent": "FinBrain/2.0"})
        with _ur.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Data API 不可用 ({url}): {str(e)}。请检查 FINBRAIN_DATA_API 配置或切回 local 模式。"}


def _get_source(key: str) -> str:
    """读取数据源配置，含校验和友好报错。key 为 stock_price/financials/industry/fund_flow"""
    env_key = f"DATA_SOURCE_{key.upper()}"
    source = os.getenv(env_key, _SOURCE_DEFAULTS.get(key, "unknown"))
    supported = _SUPPORTED_SOURCES.get(key, {})
    if source not in supported:
        known = ", ".join(f"{k}({v})" for k, v in supported.items())
        logger.warning("Unsupported data source '%s' for '%s'. Supported: %s. Falling back to default.",
                       source, key, known)
        return _SOURCE_DEFAULTS.get(key, "unknown")
    return source


def _source_error(source: str, detail: str = "") -> dict:
    """生成友好的数据源错误信息"""
    name = "未知"
    for group in _SUPPORTED_SOURCES.values():
        if source in group:
            name = group[source]
            break
    msg = f"数据源 [{source}] ({name}) 请求失败"
    if detail:
        msg += f": {detail}"
    msg += "。可在 Settings > Data Sources 中切换数据源。"
    return {"error": msg}

# ---- SSL 配置 ----
# 说明：新浪和东方财富的免费API服务器使用自签名/过期证书，
# 在Windows 11 + Python 3.12环境下TLS握手会触发
# `[SSL: UNEXPECTED_EOF_WHILE_READING]` 错误。
# 这不是我们代码的问题，是服务器端证书链不完整。
# 生产环境应使用付费数据源（Wind/Tushare Pro）替代。
# 如果合规要求严格，可改为加载服务器证书到信任链。
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# ============================================================
#  工具1：实时行情
# ============================================================

def fetch_stock_price(symbol: str) -> dict:
    """查询A股实时价格（数据源可配置，失败时回退到 DATA_SLOT）"""
    if _should_skip("stock_price"):
        result = {"error": "stock_price 已连续失败3次，本轮跳过（熔断）"}
        return _try_with_data_slots(symbol, result, "stock_price", ["price"])
    dup = _dedup_check(symbol, "stock_price")
    if dup:
        return dup
    if _DATA_MODE == "remote":
        result = _remote_fetch("api/data/stock_price", {"symbol": symbol})
        _dedup_record(symbol, "stock_price", result)
        return result

    src = _get_source("stock_price")
    if src not in ("sina", "akshare"):
        result = _source_error(src, "仅支持 sina/akshare")
        return _try_with_data_slots(symbol, result, "stock_price", ["price"])

    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        result = {"error": f"无法识别的股票代码: {symbol}"}
        return _try_with_data_slots(symbol, result, "stock_price", ["price"])

    url = f"https://hq.sinajs.cn/list={full_code}"

    # 先查缓存
    from backend import cache
    cached = cache.get("stock_price", symbol)
    if cached:
        return cached

    try:
        req = urllib.request.Request(url)
        req.add_header("Referer", "https://finance.sina.com.cn")
        req.add_header("User-Agent", _USER_AGENT)

        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            text = resp.read().decode("gbk")

        match = re.search(r'"([^"]+)"', text)
        if not match:
            result = {"error": f"未找到股票数据: {text[:100]}"}
            return _try_with_data_slots(symbol, result, "stock_price", ["price"])

        fields = match.group(1).split(",")
        if len(fields) < 32:
            result = {"error": f"返回数据字段不足: {len(fields)}"}
            return _try_with_data_slots(symbol, result, "stock_price", ["price"])

        result = {
            "name": fields[0],
            "open": fields[1],
            "yesterday_close": fields[2],
            "price": fields[3],
            "high": fields[4],
            "low": fields[5],
            "time": f"{fields[30]} {fields[31]}",
        }
        cache.set("stock_price", symbol, result)
        _record_success("stock_price")
        return result

    except urllib.error.URLError as e:
        _record_failure("stock_price")
        result = {"error": f"网络请求失败: {str(e)}"}
        return _try_with_data_slots(symbol, result, "stock_price", ["price"])
    except Exception as e:
        _record_failure("stock_price")
        result = {"error": f"查询异常: {str(e)}"}
        return _try_with_data_slots(symbol, result, "stock_price", ["price"])


# ============================================================
#  工具2：历史K线
# ============================================================

def fetch_stock_history(symbol: str, scale: int = 240, datalen: int = 30) -> dict:
    """通过新浪API查询A股历史K线数据"""
    if _DATA_MODE == "remote":
        return _remote_fetch("api/data/stock_history", {"symbol": symbol, "scale": scale, "datalen": datalen})
    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        return {"error": f"无法识别的股票代码: {symbol}"}

    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={full_code}&scale={scale}&ma=no&datalen={datalen}"
    )

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent",
                       _USER_AGENT)
        req.add_header("Referer", "https://finance.sina.com.cn")


        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            text = resp.read().decode("utf-8")
            data = json.loads(text)

        klines = []
        for item in data:
            klines.append({
                "day": item["day"],
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
                "volume": item["volume"],
            })

        return {"data": klines}

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}


# ============================================================
#  工具3：三大报表
# ============================================================

def get_financial_statements(symbol: str) -> dict:
    """获取近2年财报（年报+季报，东财 datacenter API）"""
    if _DATA_MODE == "remote":
        return _remote_fetch("api/data/financials", {"symbol": symbol})
    from backend import cache
    cached = cache.get("financial_statements", symbol)
    if cached:
        return cached
    try:
        headers = {"User-Agent":
                   _USER_AGENT}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

        def fetch_report(report_name):
            params = {
                "reportName": report_name,
                "columns": "ALL",
                "filter": f'(SECURITY_CODE="{symbol}")',
                "pageNumber": "1",
                "pageSize": "8",
                "sortColumns": "REPORT_DATE",
                "sortTypes": "-1",
                "source": "SECURITIES",
                "client": "PC",
            }
            url = base + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["result"]["data"]

        balance_data  = fetch_report("RPT_DMSK_FN_BALANCE")
        profit_data   = fetch_report("RPT_DMSK_FN_INCOME")
        cashflow_data = fetch_report("RPT_DMSK_FN_CASHFLOW")

        BALANCE_COLS = {
            "TOTAL_ASSETS": "资产总计", "TOTAL_LIABILITIES": "负债合计",
            "TOTAL_EQUITY": "股东权益", "FIXED_ASSET": "固定资产",
            "MONETARYFUNDS": "货币资金", "ACCOUNTS_RECE": "应收账款",
            "INVENTORY": "存货", "ACCOUNTS_PAYABLE": "应付账款",
            "DEBT_ASSET_RATIO": "资产负债率",
            # 有息负债（支柱五 ROIC proxy 用，字段缺失时为 None 不影响其他列）
            "SHORT_LOAN": "短期借款", "LONG_LOAN": "长期借款",
            "BOND_PAYABLE": "应付债券", "NONCURRENT_LIAB_1YEAR": "一年内到期的非流动负债",
        }
        INCOME_COLS = {
            "TOTAL_OPERATE_INCOME": "营业总收入", "OPERATE_COST": "营业成本",
            "PARENT_NETPROFIT": "归母净利润", "DEDUCT_PARENT_NETPROFIT": "扣非净利润",
            "SALE_EXPENSE": "销售费用", "MANAGE_EXPENSE": "管理费用",
            "FINANCE_EXPENSE": "财务费用", "OPERATE_PROFIT": "营业利润",
            "INCOME_TAX": "所得税",
        }
        CASHFLOW_COLS = {
            "NETCASH_OPERATE": "经营现金流净额", "NETCASH_INVEST": "投资现金流净额",
            "NETCASH_FINANCE": "筹资现金流净额", "CONSTRUCT_LONG_ASSET": "购建固定资产支付现金",
            "DEPRECIATION": "折旧摊销",
        }

        # 财报类型标注
        _REPORT_TYPE = {"001": "年报", "002": "半年报", "003": "一季报", "004": "三季报"}

        def pick(data, col_map):
            results = []
            for row in data:
                dt_code = row.get("DATE_TYPE_CODE", "")
                period = _REPORT_TYPE.get(dt_code, dt_code)
                item = {"date": row["REPORT_DATE"][:10], "报告期": period}
                for eng_key, cn_name in col_map.items():
                    item[cn_name] = row.get(eng_key, None)
                results.append(item)
            return results

        result = {
            "symbol": symbol,
            "balance":  pick(balance_data,  BALANCE_COLS),
            "profit":   pick(profit_data,   INCOME_COLS),
            "cashflow": pick(cashflow_data, CASHFLOW_COLS),
            "name": balance_data[0].get("SECURITY_NAME_ABBR", ""),
        }
        # 业绩快报datacenter(RPT_FCI_PERFORMANCEE)：快报发布次日即入库，比三大报表快4-6周。
        # 若其最新期间比利润表更新，合成headline行前置插入（扣非字段缺失，由公告快报补齐）。
        try:
            perf_rows = fetch_report("RPT_FCI_PERFORMANCEE")
            if perf_rows:
                newest = perf_rows[0]
                rd = (newest.get("REPORT_DATE") or "")[:10]
                if rd and newest.get("PARENT_NETPROFIT") is not None:
                    profit_list = result["profit"]
                    if not profit_list or profit_list[0].get("date", "") < rd:
                        _PERIOD_MAP = {"03-31": "一季报", "06-30": "半年报",
                                       "09-30": "三季报", "12-31": "年报"}
                        profit_list.insert(0, {
                            "date": rd,
                            "报告期": _PERIOD_MAP.get(rd[5:], "季报"),
                            "营业总收入": newest.get("TOTAL_OPERATE_INCOME"),
                            "归母净利润": newest.get("PARENT_NETPROFIT"),
                            "扣非净利润": None,
                            "_快报源": True,
                        })
        except Exception:
            pass  # 快源失败不影响主报表
        cache.set("financial_statements", symbol, result)
        return result

    except Exception as e:
        result = {"error": f"获取财报失败: {str(e)}"}
        return _try_with_data_slots(symbol, result, "financials", ["balance", "profit", "cashflow"])


# ============================================================
#  工具4：估值指标
# ============================================================

def get_valuation(symbol: str) -> dict:
    """获取估值数据（ROE/毛利率/净利率/EPS等）"""
    if _DATA_MODE == "remote":
        return _remote_fetch("api/data/valuation", {"symbol": symbol})
    from backend import cache
    cached = cache.get("valuation", symbol)
    if cached:
        return cached
    try:
        headers = {"User-Agent":
                   _USER_AGENT}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        params = {
            "reportName": "RPT_F10_FINANCE_MAINFINADATA",
            "columns": "ALL",
            "filter": f'(SECURITY_CODE="{symbol}")',
            "pageNumber": "1",
            "pageSize": "3",
            "sortColumns": "REPORT_DATE",
            "sortTypes": "-1",
            "source": "SECURITIES",
            "client": "PC",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        VALUATION_COLS = {
            "REPORT_DATE_NAME": "报告期", "ROEJQ": "ROE(%)",
            "XSMLL": "毛利率(%)", "XSJLL": "净利率(%)",
            "EPSJB": "每股收益", "BPS": "每股净资产",
            "TOTAL_SHARE": "总股本", "PARENTNETPROFIT": "归母净利润",
            "TOTALOPERATEREVE": "营业总收入", "ZCFZL": "资产负债率(%)",
        }

        results = []
        for row in data["result"]["data"]:
            item = {"日期": row["REPORT_DATE"][:10]}
            for eng_key, cn_name in VALUATION_COLS.items():
                item[cn_name] = row.get(eng_key, None)
            results.append(item)

        result = {
            "symbol": symbol,
            "data": results,
            "name": data["result"]["data"][0].get("SECURITY_NAME_ABBR", ""),
        }
        cache.set("valuation", symbol, result)
        return result

    except Exception as e:
        result = {"error": f"获取估值失败: {str(e)}"}
        return _try_with_data_slots(symbol, result, "valuation", ["data"])


# ============================================================
#  工具4c：报告期推算（财报期次纪律配套）
# ============================================================

def expected_recent_periods(ref_date=None, count: int = 3) -> list:
    """按A股披露节奏，以系统时间动态推算最近的 count 个应已披露的报告期。

    披露截止：一季报/年报 次年4月30日，半年报 8月31日，三季报 10月31日。
    返回 [(期末日期str, 期次标签)]，最新在前。
    例（ref=2026-07-26）: [("2026-03-31","一季报"),("2025-12-31","年报"),("2025-09-30","三季报")]
    """
    from datetime import date as _date
    ref = ref_date or _date.today()
    cands = []
    for y in (ref.year, ref.year - 1, ref.year - 2):
        cands += [
            (_date(y, 12, 31), "年报", _date(y + 1, 4, 30)),
            (_date(y, 9, 30), "三季报", _date(y, 10, 31)),
            (_date(y, 6, 30), "半年报", _date(y, 8, 31)),
            (_date(y, 3, 31), "一季报", _date(y, 4, 30)),
        ]
    avail = [(end, label) for end, label, ddl in cands if ddl <= ref and end < ref]
    avail.sort(key=lambda x: x[0], reverse=True)
    return [(d.isoformat(), lbl) for d, lbl in avail[:count]]


# ============================================================
#  工具4b：主营构成（利润引擎解剖，分类引擎修正方法 支柱一/二）
# ============================================================

def get_business_segments(symbol: str) -> dict:
    """获取主营构成（按产品分类，东财 via akshare）：最新两期板块收入/毛利/毛利率。

    返回 {"报告期": "2025-12-31", "segments": [{名称,收入,收入占比,毛利额,毛利占比,毛利率}],
          "前期": [...], "前期报告期": str}。
    板块净利润不可得，毛利额占比为"利润引擎占比"的免费源最佳近似（口径见文档）。
    失败返回 {"error": ...}，不抛异常。"""
    from backend import cache
    cached = cache.get("business_segments", symbol)
    if cached:
        return cached
    try:
        import akshare as _ak
        prefix = ("SH" if symbol.startswith(("60", "68")) else
                  "BJ" if symbol.startswith(("4", "8", "92")) else "SZ")
        df = _ak.stock_zygc_em(symbol=f"{prefix}{symbol}")
        if df is None or len(df) == 0:
            return {"error": "主营构成无数据"}
        df = df[df["分类类型"] == "按产品分类"]
        if len(df) == 0:
            return {"error": "主营构成(按产品)无数据"}
        periods = sorted(df["报告日期"].astype(str).unique().tolist(), reverse=True)
        # 前期必须同期次（年报↔年报，半年报↔半年报）：
        # 否则年报(全年)对半年报(半年)的收入同比会出现 ~+100% 的伪增长
        _md = periods[0][5:]
        _pool = [p for p in periods if p[5:] == _md and p != periods[0]]
        prev_period = _pool[0] if _pool else (periods[1] if len(periods) > 1 else None)

        def _rows(period):
            sub = df[df["报告日期"].astype(str) == period]
            segs = []
            for _, r in sub.iterrows():
                name = str(r["主营构成"]).strip()
                # "其中:xx" 是主板块的子项拆分，"其他/其他(补充)" 单列但不参与引擎判定
                segs.append({
                    "名称": name,
                    "收入": float(r["主营收入"] or 0),
                    "收入占比": float(r["收入比例"] or 0),
                    "毛利额": float(r["主营利润"] or 0),
                    "毛利占比": float(r["利润比例"] or 0),
                    "毛利率": float(r["毛利率"]) if r["毛利率"] is not None else None,
                    "_子项": name.startswith(("其中:", "其中：")),
                    "_其他": name in ("其他", "其他(补充)", "其他（补充）"),
                })
            return segs

        result = {"报告期": periods[0], "segments": _rows(periods[0]),
                  "前期报告期": prev_period,
                  "前期": _rows(prev_period) if prev_period else []}
        cache.set("business_segments", symbol, result)
        return result
    except Exception as e:
        return {"error": f"获取主营构成失败: {str(e)}"}



# ============================================================
#  工具5：行业信息
# ============================================================

def get_industry_info(symbol: str) -> dict:
    """获取个股所属行业 + 行业指数近期表现"""
    from backend import cache
    cached = cache.get("industry_info", symbol)
    if cached:
        return cached
    try:
        headers = {"User-Agent":
                   _USER_AGENT}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        params = {
            "reportName": "RPT_DMSK_FN_BALANCE",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,INDUSTRY_NAME,INDUSTRY_CODE",
            "filter": f'(SECURITY_CODE="{symbol}")',
            "pageNumber": "1",
            "pageSize": "1",
            "source": "SECURITIES",
            "client": "PC",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        row = data["result"]["data"][0]
        industry_name = row.get("INDUSTRY_NAME", "")

        # 行业指数（THS，数据截至2024年初）
        index_data = []
        if industry_name:
            try:
                df = ak.stock_board_industry_index_ths(symbol=industry_name)
                recent = df.tail(6)[["日期", "收盘价"]]
                prev_close = None
                for _, r in recent.iterrows():
                    close = r["收盘价"]
                    item = {"日期": str(r["日期"])[:10], "收盘": close}
                    if prev_close and prev_close != 0:
                        item["涨跌幅"] = f"{(close - prev_close) / prev_close * 100:+.2f}%"
                    else:
                        item["涨跌幅"] = "-"
                    prev_close = close
                    index_data.append(item)
                if index_data:
                    index_data = index_data[-5:]
            except Exception:
                pass

        result = {
            "symbol": symbol,
            "name": row.get("SECURITY_NAME_ABBR", ""),
            "industry_name": industry_name,
            "industry_code": row.get("INDUSTRY_CODE", ""),
            "index_trend": index_data,
        }
        cache.set("industry_info", symbol, result)
        return result

    except Exception as e:
        result = {"error": f"获取行业信息失败: {str(e)}"}
        return _try_with_data_slots(symbol, result, "industry_info", ["industry_name"])


# ============================================================
#  工具6：全市场扫描
# ============================================================

def screen_stocks(max_pe: float = 30, max_pb: float = 5,
                  min_mktcap: float = 20, top_n: int = 30) -> dict:
    """全市场扫描，按PE升序返回低估值A股。
       参数: max_pe=市盈率上限, max_pb=市净率上限, min_mktcap=市值下限(亿)"""
    try:
        headers = {"User-Agent":
                   _USER_AGENT}

        # 新浪全市场行情接口（分4页覆盖沪深A股）
        all_stocks = []
        seen = set()

        for page in range(1, 5):
            url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/"
                   f"json_v2.php/Market_Center.getHQNodeData?"
                   f"page={page}&num=2000&sort=code&asc=1&node=hs_a")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("gbk"))

            for s in data:
                code = s.get("code", "")
                name = s.get("name", "")

                if code in seen:
                    continue
                if "ST" in name or "退" in name or "N" in name:
                    continue

                try:
                    pe = float(s.get("per", 0) or 0)
                    pb = float(s.get("pb", 0) or 0)
                    mktcap = float(s.get("mktcap", 0) or 0)
                    change = float(s.get("changepercent", 0) or 0)
                except (ValueError, TypeError):
                    continue

                # 过滤条件
                if pe <= 0 or pe > max_pe:
                    continue
                if pb <= 0 or pb > max_pb:
                    continue
                if mktcap < min_mktcap:
                    continue

                seen.add(code)
                all_stocks.append({
                    "代码": code,
                    "名称": name,
                    "PE": round(pe, 1),
                    "PB": round(pb, 2),
                    "市值(亿)": round(mktcap, 0),
                    "涨跌幅": round(change, 2),
                })

        # 按PE升序
        all_stocks.sort(key=lambda x: x["PE"])
        top = all_stocks[:top_n]

        # 格式化为对齐文本表格
        keys = ["代码", "名称", "PE", "PB", "市值(亿)", "涨跌幅"]
        header = ["代码", "名称", "  PE", "  PB", "市值(亿)", " 涨跌%"]
        rows = [[str(r[k]) if k != "名称" else r[k] for k in keys] for r in top]

        # 计算每列最大宽度
        col_widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

        def align(val, width, right=False):
            return f"{val:>{width}}" if right else f"{val:<{width}}"

        lines = [f"筛选: PE 0-{max_pe}, PB<{max_pb}, 市值>{min_mktcap}亿 | 共{len(top)}只"]
        lines.append("  ".join(align(h, w, i >= 2) for i, (h, w) in enumerate(zip(header, col_widths))))
        lines.append("  ".join("-" * w for w in col_widths))
        for row in rows:
            lines.append("  ".join(align(v, w, i >= 2) for i, (v, w) in enumerate(zip(row, col_widths))))

        return {"text": "\n".join(lines)}

    except Exception as e:
        return {"error": f"全市场扫描失败: {str(e)}"}


# ============================================================
#  工具6：资金流向
# ============================================================

def _get_ths_headers() -> dict:
    """生成同花顺接口需要的 hexin-v 鉴权头"""
    from py_mini_racer import MiniRacer
    import os as _os
    ths_js = _os.path.join(
        _os.path.dirname(ak.__file__), "stock_feature", "ths.js")
    with open(ths_js) as f:
        js = MiniRacer()
        js.eval(f.read())
    return {
        "hexin-v": js.call("v"),
        "Referer": "http://data.10jqka.com.cn/funds/hyzjl/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Requested-With": "XMLHttpRequest",
    }


def get_fund_flow(symbol: str) -> dict:
    """获取个股当日资金流向（主力/超大单/大单/中单/小单净额）"""
    try:
        import requests
        from bs4 import BeautifulSoup as bs

        headers = _get_ths_headers()

        # 扫前5页排名，搜目标股票
        for page in range(1, 6):
            url = (f"http://data.10jqka.com.cn/funds/ggzjl/"
                   f"field/zdf/order/desc/page/{page}/ajax/1/free/1/")
            resp = requests.get(url, headers=headers, timeout=10)
            soup = bs(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                continue

            for row in table.find_all("tr")[1:]:
                cols = [td.text.strip() for td in row.find_all("td")]
                if len(cols) < 10:
                    continue
                if cols[1] == symbol:
                    return {
                        "代码": symbol,
                        "名称": cols[2],
                        "最新价": cols[3],
                        "涨跌幅": cols[4],
                        "换手率": cols[5],
                        "流入(元)": cols[6],
                        "流出(元)": cols[7],
                        "净额(元)": cols[8],
                        "成交额(元)": cols[9],
                    }

        result = {"error": f"未在资金流排名前5页找到 {symbol}"}
        return _try_with_data_slots(symbol, result, "fund_flow", ["净额(元)"])

    except Exception as e:
        result = {"error": f"资金流查询失败: {str(e)}"}
        return _try_with_data_slots(symbol, result, "fund_flow", ["净额(元)"])


# ============================================================
#  工具7：格式化报告（Reporter 用，纯函数，不需要 LLM）
# ============================================================

def _align_col(v, w, right=False):
    """对齐辅助：数字右对齐，文本左对齐"""
    s = str(v) if v is not None else "-"
    return f"{s:>{w}}" if right else f"{s:<{w}}"


def _format_compare_section(compare: dict) -> str:
    """将对比分析 dict 渲染为独立的对比表区块（不包含单股评分卡）。
    供 reporter_node 提取多股票 JSON 中最后一支的"对比分析"字段时使用。
    """
    lines = []
    lines.append("")
    lines.append("  " + "=" * 58)
    lines.append(f"  [板块对比] {compare.get('板块', '')}")
    lines.append("  " + "=" * 58)

    # 财报对比表
    fin_table = compare.get("财报对比表", {})
    if fin_table:
        indicators = fin_table.get("指标列表", [])
        stocks = fin_table.get("股票数据", [])
        if indicators and stocks:
            lines.append("")
            lines.append("  [财报核心数据对比]")
            header_cols = ["指标"] + [s.get("名称", "?") for s in stocks]
            col_widths = [max(12, max(len(str(s.get(h, ""))) for s in stocks)) + 2 for h in indicators]
            col_widths.insert(0, max(len(h) for h in header_cols) + 2)

            header_line = "  " + "".join(_align_col(h, w) for h, w in zip(header_cols, col_widths))
            lines.append(header_line)
            lines.append("  " + "".join("-" * w for w in col_widths))

            for idx in indicators:
                row = [_align_col(idx, col_widths[0])]
                for j, s in enumerate(stocks):
                    row.append(_align_col(s.get(idx, "-"), col_widths[j+1], True))
                lines.append("  " + "".join(row))

    # 估值对比表
    val_table = compare.get("估值对比表", {})
    if val_table:
        v_indicators = val_table.get("指标列表", [])
        v_stocks = val_table.get("股票数据", [])
        if v_indicators and v_stocks:
            lines.append("")
            lines.append("  [估值对比]")
            v_header = ["指标"] + [s.get("名称", "?") for s in v_stocks]
            v_widths = [max(12, max(len(str(s.get(h, ""))) for s in v_stocks)) + 2 for h in v_indicators]
            v_widths.insert(0, max(len(h) for h in v_header) + 2)

            lines.append("  " + "".join(_align_col(h, w) for h, w in zip(v_header, v_widths)))
            lines.append("  " + "".join("-" * w for w in v_widths))
            for idx in v_indicators:
                row = [_align_col(idx, v_widths[0])]
                for j, s in enumerate(v_stocks):
                    row.append(_align_col(s.get(idx, "-"), v_widths[j+1], True))
                lines.append("  " + "".join(row))

    # 差异解读
    diffs = compare.get("差异解读", [])
    if diffs:
        lines.append("")
        lines.append("  [差异解读]")
        for d in diffs:
            lines.append(f"    {d}")

    # 一句话总结
    summary = compare.get("一句话总结", {})
    if summary:
        lines.append("")
        for name, pos in summary.items():
            lines.append(f"    {name}: {pos}")

    rank = compare.get("综合排名", "")
    if rank:
        lines.append("")
        lines.append(f"  [综合排名] {rank}")

    return "\n".join(lines)


def format_report(analysis: dict) -> str:
    """将 Analyst 的结构化 JSON 转换为对齐文本表格。

    期望输入格式:
    {
      "代码": "300502", "名称": "新易盛",
      "评分": {
        "盈利能力": {"得分": 9, "依据": "ROE 72.75%, 毛利率47.8%, 净利率38.5%"},
        "成长性":   {"得分": 10, "依据": "营收+187%, 净利润+236%, 两季增速>30%"},
        "财务健康": {"得分": 8, "依据": "负债率30.2%, 经营现金流77亿, 利润含金量高"},
        "估值合理": {"得分": 6, "依据": "PE 53倍偏高, PEG约0.2, 行业高景气可接受"},
        "行业前景": {"得分": 9, "依据": "AI光模块龙头, 全球算力基建持续3-5年"},
        "资金认可": {"得分": 8, "依据": "当日净流入19.18亿, 主力认可度高"}
      },
      "亮点": ["ROE 72.75% A股顶尖", "2年利润14倍", "现金流远超净利润"],
      "风险": ["客户集中北美地缘风险", "PE 53倍波动大", "竞争格局变化"],
      "操作建议": "等待回调10-15%至450附近轻仓试错",
      "止损": "成本价-8% 或跌破60日线",
      "结论": "基本面A+, 估值中等, 回调是买点, AI算力最受益标的"
    }
    """
    try:
        lines = []
        code = analysis.get("代码", "?")
        name = analysis.get("名称", "?")
        current_price = (
            analysis.get("当前股价", 0)
            or analysis.get("投资评级", {}).get("当前价格", 0)
            or analysis.get("_fd_ctx", {}).get("stock_price", 0)
        )
        # 兜底：从操作建议/估值水位/财报数据中提取价格
        if not current_price:
            import re as _re_price
            # 操作建议文本通常含"当前X元"或"≤X元(当前Y元)"
            _adv = str(analysis.get("操作建议", ""))
            _m = _re_price.search(r'当前\s*([\d.]+)\s*元', _adv)
            if _m:
                current_price = float(_m.group(1))
        if not current_price:
            # 从估值水位尝试反推：PE和EPS如果都有，价格=PE×EPS
            _vw = analysis.get("估值水位", {})
            _pe = _vw.get("PE") if isinstance(_vw, dict) else None
            if _pe:
                try:
                    _pe = float(str(_pe).replace("倍", ""))
                    # 从_fd_ctx获取EPS
                    _eps = analysis.get("_fd_ctx", {}).get("eps_ttm", 0)
                    if _eps > 0 and _pe > 0:
                        current_price = round(_pe * _eps, 2)
                except Exception:
                    pass

        lines.append(f"=" * 64)
        lines.append(f"  FinBrain 投资研究: {name} ({code})")
        lines.append(f"=" * 64)

        # === 第一部分：投资结论先行 ===
        # 投资逻辑链
        logic_chain = analysis.get("投资逻辑链", "")
        if logic_chain:
            lines.append("")
            lines.append(f"  [投资逻辑] {logic_chain}")

        # 投资评级(代码强制)
        rating = analysis.get("投资评级", {})
        if rating:
            lines.append("")
            level = rating.get("评级", "?")
            fair = rating.get("合理价值", "?")
            margin = (rating.get("实际安全边际") or rating.get("安全边际要求")
                      or rating.get("安全边际") or "?")
            buy_zone = rating.get("买入区间", "?")
            gap = rating.get("估值差距", "")
            weighted = rating.get("加权总分", "")
            conf = rating.get("置信度", "")
            level_icon = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(level, "⚪")
            price_str = f"当前价{float(current_price):.2f}元" if current_price else "当前价: 数据缺失"
            has_divergence = bool(analysis.get("框架分歧", ""))
            if has_divergence:
                lines.append(f"  [投资决策] {level_icon} {level}(量化锚点) / 趋势框架→见下方分歧  {price_str}  合理价值: {fair}  安全边际: {margin}")
            else:
                lines.append(f"  [投资决策] {level_icon} {level}  {price_str}  合理价值: {fair}  安全边际: {margin}")
            if gap: lines.append(f"    估值差距: {gap}")
            _margin_req = rating.get('安全边际要求') or '?'
            lines.append(f"    安全买入价: {buy_zone}  (需{_margin_req}安全边际)" if "不适用" not in str(buy_zone) else f"    安全买入价: {buy_zone}")
            # 双锚点：情景加权为主 + PE乘数法为参考地板
            _pe_ref = rating.get("合理价值(PE乘数法)") if isinstance(rating, dict) else None
            _val_note = rating.get("估值方法说明", "") if isinstance(rating, dict) else ""
            if _pe_ref is not None:
                lines.append(f"    📊 估值方法: 主锚点=情景概率加权（机构常用多情景DCF），参考地板=PE乘数法({_pe_ref}元)")
            _fair_range = rating.get("合理区间", "") if isinstance(rating, dict) else ""
            if _fair_range:
                lines.append(f"    合理区间（三锚点）: {_fair_range}")
            # 估值差距解读（合理价值仅为现价零头时）：为什么差这么多 + 趋势参考买入价
            _gap_expl = analysis.get("估值差距解读", [])
            if _gap_expl:
                lines.append(f"    ⚠️ 估值差距解读（合理价值为何远低于现价）:")
                for _g in _gap_expl[:6]:
                    lines.append(f"      - {_g}")
            _tb = analysis.get("趋势参考买入价")
            if _tb:
                lines.append(f"    趋势参考买入价: ≈{_tb}元（近期技术支撑×1.05，趋势框架入场锚，"
                             "与价值锚安全买入价并列，按自身风格选用）")
            if weighted: lines.append(f"    加权总分: {weighted}/100  置信度: {conf}")
            lines.append(f"    * 估值基于财报数据计算，非实时定价。不构成买卖建议。")

        # ---- 市场情绪参考（代码计算，仅作决策参考，不参与估值） ----
        sentiment = analysis.get("市场情绪", {})
        if sentiment and isinstance(sentiment, dict):
            lines.append("")
            lines.append(f"  [市场情绪参考] 仅作节奏参考，不纳入估值")
            score = sentiment.get("综合情绪得分", 0)
            label = sentiment.get("情绪标签", "")
            breadth = sentiment.get("市场广度", {})
            stock_chg = sentiment.get("个股涨跌", {})
            heat = sentiment.get("短线热度", {})
            suggestion = sentiment.get("对操作建议", "")
            lines.append(f"    综合情绪: {score:+.2f} ({label})")
            lines.append(f"    市场广度: 上涨比例 {breadth.get('上涨比例', 'N/A')}")
            lines.append(f"    个股涨跌: {stock_chg.get('说明', 'N/A')}")
            lines.append(f"    短线热度: {heat.get('备注', '无')}")
            if suggestion: lines.append(f"    情绪建议: {suggestion}")
            lines.append(f"    * 情绪指标基于当日行情数据，可能随市场波动快速变化，请勿作为独立决策依据。")

        # 估值方法
        val_method = analysis.get("估值方法", "")
        if val_method:
            lines.append("")
            lines.append(f"  [估值方法] {val_method}")

        # 估值框架（Valuation Agent 输出）
        val_framework = analysis.get("估值框架", "")
        if val_framework:
            lines.append(f"  [估值框架] {val_framework}")

        # 估值明细（代码计算链，透明化）
        val_chain = rating.get("估值明细", {}) if isinstance(rating, dict) else {}
        if val_chain and val_chain.get("公式"):
            lines.append(f"  [估值明细] {val_chain['公式']}")
            # 因子行键驱动渲染（兼容 quality×growth 路径与支柱三乘数矩阵路径）
            _factor_parts = []
            for _k in ("行业PE中枢", "财务质量乘数", "成长溢价", "乘数矩阵", "EPS(TTM)"):
                if _k in val_chain:
                    _v = val_chain[_k]
                    _factor_parts.append(f"{_k}={_v}{'元' if _k == 'EPS(TTM)' else ''}")
            if _factor_parts:
                lines.append("    " + " | ".join(_factor_parts))
            if val_chain.get("乘数矩阵说明"):
                lines.append(f"    📐 {val_chain['乘数矩阵说明']}")
            if val_chain.get("前瞻说明"):
                lines.append(f"    ⚠️ {val_chain['前瞻说明']}")
            if val_chain.get("PB地板"):
                lines.append(f"    ⚠️ {val_chain['PB地板']}")

        # === 第二部分：公司画像和竞争优势 ===
        profile = analysis.get("公司画像", {})
        if profile:
            lines.append("")
            lines.append("  [公司画像]")
            biz = profile.get("主营业务", "")
            if biz: lines.append(f"    主营业务: {biz}")
            ctype = profile.get("公司类型", "")
            lifecycle = profile.get("生命周期", "")
            if ctype or lifecycle:
                lines.append(f"    类型: {ctype} | 生命周期: {lifecycle}" if ctype and lifecycle else
                            f"    类型: {ctype or lifecycle}")

        # ---- 竞争优势 ----
        moat = analysis.get("竞争优势", {})
        if moat:
            lines.append("")
            lines.append("  [竞争优势]")
            core = moat.get("核心资产", "")
            if core: lines.append(f"    核心资产: {core}")
            moat_src = moat.get("护城河来源", moat.get("护城河类型", []))
            if moat_src: lines.append(f"    壁垒来源: {', '.join(moat_src)}")
            diff = moat.get("复制难度", "")
            dur = moat.get("持续时间", moat.get("可持续性", ""))
            if diff or dur: lines.append(f"    复制难度: {diff} | 持续时间: {dur}")
            landscape = moat.get("竞争格局", "")
            if landscape: lines.append(f"    竞争格局: {landscape}")
            gm_attr = moat.get("毛利率归因", "")
            if gm_attr: lines.append(f"    毛利率归因: {gm_attr}")

        # ---- 利润引擎（支柱一/二：主营构成解剖，毛利占比≈利润贡献）----
        engines = analysis.get("_engines", {})
        if isinstance(engines, dict) and engines.get("板块"):
            lines.append("")
            lines.append(f"  [利润引擎] 主营构成解剖（{engines.get('报告期','?')}，毛利占比≈利润引擎占比）")
            for seg in engines["板块"][:6]:
                gm = f"{seg['毛利率']*100:.1f}%" if seg.get("毛利率") is not None else "-"
                rg = f"{seg['收入同比']*100:+.0f}%" if seg.get("收入同比") is not None else "-"
                lines.append(f"    {seg['名称']}: 收入占比{seg['收入占比']*100:.1f}% | "
                             f"毛利占比{seg['毛利占比']*100:.1f}% | 毛利率{gm} | 同比{rg} | {seg['类型']}")
            if engines.get("锚定引擎"):
                lines.append(f"    ⚓ 单一引擎毛利占比>40%：估值锚向「{engines['锚定引擎']}」倾斜（支柱一）")
            for _t in (engines.get("SOTP触发") or []):
                lines.append(f"    🔀 SOTP强制隔离阀: {_t}")
            if engines.get("SOTP参考估值"):
                lines.append(f"    SOTP参考估值: {engines['SOTP参考估值']}元/股（毛利占比分配净利近似，仅作对照）")

        # ---- 评分表格 ----
        scores = analysis.get("评分", {})
        if scores:
            lines.append("")
            lines.append("  [评分卡]  满分10分")
            lines.append(f"  {'维度':<8}  {'得分':>4}  {'评级':<6}  依据")
            lines.append(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*44}")

            def grade(s):
                if s >= 9: return "S"
                if s >= 7: return "A"
                if s >= 5: return "B"
                return "C"

            total = 0
            max_total = 0
            # 复合指标（非0-10分量表），展示但不参与维度合计
            _AGGREGATE_KEYS = {"加权总分", "综合评级", "置信度"}
            for dim, info in scores.items():
                if not isinstance(info, dict):
                    # 评分维度数据缺失或格式错误，防御性跳过
                    lines.append(f"  {dim:<8}  {'N/A':>4}  {'-':<6}  数据缺失")
                    continue
                s_val = info.get("得分")
                reason = info.get("依据", "")
                # 防御：LLM 可能输出字符串得分，转换为数值
                try:
                    if isinstance(s_val, str):
                        s_val = float(s_val.replace("分", "").strip()) if s_val.strip() else None
                except (ValueError, AttributeError):
                    s_val = None
                is_aggregate = dim in _AGGREGATE_KEYS or (isinstance(s_val, (int, float)) and s_val > 10)
                if s_val is None:
                    # 数据缺失，不参与计分
                    lines.append(f"  {dim:<8}  {'N/A':>4}  {'-':<6}  {reason}")
                else:
                    g = grade(s_val) if not is_aggregate else "-"
                    cf_label = info.get("现金流标签", "") if isinstance(info, dict) else ""
                    display_reason = f"{reason} {cf_label}" if cf_label else reason
                    lines.append(f"  {dim:<8}  {s_val:>4}  {g:<6}  {display_reason}")
                    if not is_aggregate:
                        total += s_val
                        max_total += 10

            lines.append(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*44}")
            if max_total > 0:
                lines.append(f"  {'合计':<8}  {total:>4}/{max_total}   {'':<6}  " +
                             f"{'S:>=9 A:>=7 B:>=5 C:<5'}")
                lines.append(f"  * 合计为各维度简单加总(满分{max_total})；加权总分(满分100)按公司类型动态权重×10计算，"
                             f"两者口径不同，数值不可直接比较。")

        # ---- 亮点 ----
        highlights = analysis.get("亮点", [])
        if highlights:
            lines.append("")
            lines.append("  [亮点]")
            for h in highlights:
                lines.append(f"    + {h}")

        # ---- 风险 ----
        risks = analysis.get("风险", [])
        if risks:
            lines.append("")
            lines.append("  [风险]")
            for r in risks:
                lines.append(f"    - {r}")

        # ---- 业绩驱动力 ----
        driver = analysis.get("业绩驱动力", "")
        if driver:
            lines.append("")
            lines.append(f"  [业绩驱动力] {driver}")

        # ---- 关键信号 ----
        signals = analysis.get("关键信号", [])
        if signals:
            lines.append("")
            lines.append("  [关键信号]")
            for s in signals:
                name = s.get("信号", "")
                data = s.get("数据", "")
                note = s.get("解读", "")
                lines.append(f"    {name}: {data}")
                if note:
                    lines.append(f"      -> {note}")

        # ---- 估值水位 ----
        val_level = analysis.get("估值水位", {})
        if val_level:
            pe = val_level.get("PE", "-")
            pb = val_level.get("PB", "-")
            mkt = val_level.get("市值", "-")
            fwd_pe = val_level.get("前瞻PE", "")
            parts = [f"PE:{pe}", f"PB:{pb}", f"市值:{mkt}"]
            if fwd_pe: parts.append(f"前瞻PE:{fwd_pe}")
            lines.append("")
            lines.append(f"  [估值水位] {' '.join(parts)}")
            fwd_note = analysis.get("_fwd_pe_note", "")
            if fwd_note:
                lines.append(f"    ⚠️ {fwd_note}")
            # A1：历史 PE band（近2年TTM PE区间）
            if analysis.get("_pe_band"):
                lines.append(f"    📏 {analysis['_pe_band']}")
        lines.append(f"    * 基于最新财报数据计算，非实时行情。日内股价波动会导致PE/PB/市值变化。请以交易软件实时数据为准。")

        # ---- 可比公司估值对比（A2）----
        comp = analysis.get("可比公司对比", {})
        if isinstance(comp, dict) and comp.get("列表"):
            lines.append("")
            lines.append("  [可比公司估值对比]（东财实时 PE(TTM)/PB）")
            for r in comp["列表"]:
                pb_txt = f" | PB {r['PB']}" if r.get("PB") else ""
                lines.append(f"    {r['名称']}({r['代码']}): PE {r['PE']}{pb_txt}")
            if comp.get("可比PE均值") and comp.get("本股PE"):
                prem = comp["本股PE"] / comp["可比PE均值"] if comp["可比PE均值"] > 0 else 0
                lines.append(f"    可比PE均值: {comp['可比PE均值']} 倍 | 本股PE: {comp['本股PE']} 倍"
                             f" → 相对可比溢价 {prem:.1f} 倍（均值已剔除亏损股负PE）")

        # ---- 一致预期前瞻（A3：机构多年EPS一致预期/无覆盖降级为本报告情景）----
        fwd = analysis.get("一致预期前瞻", {})
        if isinstance(fwd, dict) and fwd.get("rows"):
            lines.append("")
            lines.append(f"  [一致预期前瞻]（{fwd.get('来源','')}）")
            lines.append(f"    {'年份':<14}{'EPS':>7}{'增速':>8}{'现价PE':>8}{'机构数':>6}")
            for r in fwd["rows"]:
                g = f"{r['增速']:+.0f}%" if r.get("增速") is not None else "-"
                pe_txt = str(r["现价PE"]) if r.get("现价PE") else "-"
                inst = str(r.get("机构数") or "-")
                lines.append(f"    {r['年份']:<14}{r['EPS']:>7}{g:>8}{pe_txt:>8}{inst:>6}")
            if fwd.get("基准年EPS"):
                lines.append(f"    ※ 增速以最近年报 EPS {fwd['基准年EPS']} 元为基数；现价PE=当前价÷预期EPS")
            else:
                lines.append(f"    ※ 无机构一致预期覆盖（次新/冷门股），前瞻以本报告情景为准")

        # ---- 机构共识 ----
        consensus = analysis.get("机构共识", {})
        if consensus:
            tp = consensus.get("目标价", {})
            net = consensus.get("净利润预测", {})
            ratings = consensus.get("评级分布", {})
            has_target = bool(tp.get("平均"))
            has_net = bool(net.get("2026"))
            has_rating = any(ratings.get(k) for k in ["买入","增持","中性","减持"])
            if has_target or has_net or has_rating:
                lines.append("")
                lines.append("  [机构共识]")
            if has_target:
                tp_str = f"目标价: 平均{tp['平均']}元"
                if tp.get("最高"): tp_str += f" 最高{tp['最高']}元"
                if tp.get("最低"): tp_str += f" 最低{tp['最低']}元"
                if tp.get("机构数"): tp_str += f" ({tp['机构数']}家机构)"
                lines.append(f"    {tp_str}")
            if has_net:
                net_str = ""
                for yr in sorted([k for k in net if k.isdigit()])[:3]:
                    val = net.get(yr)
                    if val:
                        net_str += f" {yr}年EPS一致预期{val}元"
                if net_str:
                    inst_str = f" ({net['机构数']}家机构)" if net.get("机构数") else ""
                    lines.append(f"    净利润预测:{net_str}{inst_str}")
            if has_rating:
                r_str = " ".join(f"{k}{ratings[k]}份" for k in ["买入","增持","中性","减持"] if ratings.get(k))
                lines.append(f"    评级分布: {r_str}")

        # ---- 情景估值 ----
        scenarios = analysis.get("情景估值", {})
        if scenarios:
            lines.append("")
            lines.append("  [情景估值]")
            for scenario, info in [("悲观", "🔴"), ("基准", "🟡"), ("乐观", "🟢")]:
                s = scenarios.get(scenario, {})
                if s:
                    price = s.get("价格", "?")
                    if isinstance(price, (int, float)):
                        price = f"{price:.2f}".rstrip("0").rstrip(".")
                    prob = s.get("概率", "?")
                    assumption = str(s.get("假设", "")).strip()
                    # 结构化 EPS/PE 字段（代码已校验 价格=EPS×PE）
                    eps_pe = ""
                    if s.get("EPS") is not None and s.get("PE") is not None:
                        eps_pe = f" [EPS={s['EPS']}×PE={s['PE']}]"
                    if assumption and not assumption.startswith("—"):
                        lines.append(f"    {info} {scenario}({prob}): {price}元{eps_pe} — {assumption}")
                    else:
                        lines.append(f"    {info} {scenario}({prob}): {price}元{eps_pe}")
            # 概率加权价值（代码重算）
            wv = scenarios.get("概率加权价值")
            if wv is not None:
                lines.append(f"    📊 概率加权价值: {wv}元")
            # 情景vs合理价值对齐检查：悲观情景价超过合理价值2倍时追加注释
            try:
                rating = analysis.get("投资评级", {}) if isinstance(analysis, dict) else {}
                fv = float(rating.get("合理价值", 0)) if isinstance(rating, dict) else 0
                pess = scenarios.get("悲观", {}) if isinstance(scenarios, dict) else {}
                pess_price = float(pess.get("价格", 0)) if isinstance(pess, dict) else 0
                if fv > 0 and pess_price > fv * 2:
                    lines.append(f"    * 情景估值由趋势研判生成，最低情景({pess_price}元)仍显著高于量化合理价值({fv:.0f}元)，"
                                 f"反映了市场情绪和成长预期——非基本面锚点，仅供参考。")
                elif fv > 0 and pess_price > fv:
                    lines.append(f"    * 注意: 情景悲观价({pess_price}元)高于量化合理价值({fv:.1f}元)——"
                                 f"两套框架锚点不同(趋势情景中枢 vs 保守PE锚)，差异解读见[框架分歧]段落。")
            except: pass

        # ---- 对比分析 ----
        compare = analysis.get("对比分析", {})
        if compare:
            lines.append(_format_compare_section(compare))

        # ---- 监测表（A5：风险触发器+观察窗口，合并代码日历/解禁/预告兑现与LLM证伪/观察）----
        monitor = analysis.get("_monitor", [])
        watch = analysis.get("观察指标", [])
        if monitor:
            lines.append("")
            lines.append("  [监测表] 风险触发器与观察窗口")
            lines.append(f"    {'观察项':<24}{'触发器/阈值':<30}{'窗口':<16}{'来源':<5}")
            for m in monitor:
                lines.append(f"    {str(m.get('观察项',''))[:22]:<24}"
                             f"{str(m.get('触发器',''))[:28]:<30}"
                             f"{str(m.get('窗口',''))[:14]:<16}"
                             f"{str(m.get('来源','')):<5}")
        elif watch:
            # 监测表未生成时回退旧版观察指标段
            lines.append("")
            lines.append("  [观察指标]")
            for w in watch:
                lines.append(f"    - {w}")

        # ---- 股本与募资信息（FIX-03：按事件类型渲染，IPO≠定增；代码修正痕迹，供审计参考）----
        gb_info = analysis.get("股本与募资信息", {})
        if isinstance(gb_info, dict) and gb_info.get("文本"):
            lines.append("")
            lines.append(f"  [股本与募资信息] {gb_info['文本']}")
        else:
            # 兼容旧键（历史 item 结构）
            zj_info = analysis.get("定增信息", {})
            if isinstance(zj_info, dict) and zj_info.get("摊薄调整系数"):
                lines.append("")
                lines.append(f"  [股本与募资信息] 代码已自动修正: 发行{zj_info.get('发行股数','?')}, "
                             f"摊薄{zj_info.get('摊薄比例','?')}, "
                             f"调整系数{zj_info.get('摊薄调整系数','?')}")
                if zj_info.get("募资金额"):
                    lines[-1] += f", 募资{zj_info['募资金额']}"
                lines[-1] += f" — {zj_info.get('说明','')}"

        # ---- 近期公告 ----
        announcements = analysis.get("公告", {}).get("列表", []) if isinstance(analysis.get("公告"), dict) else []
        if announcements:
            lines.append("")
            lines.append("  [近期关键公告]")
            reds = [a for a in announcements if a.get("级别") == "🔴"]
            yellows = [a for a in announcements if a.get("级别") == "🟡"]
            # 🔴 最高优先级
            for a in reds[:8]:
                hint = a.get("提示", "")
                lines.append(f"    🔴 {a.get('日期','')} {a.get('标题','')[:70]}")
                if hint: lines.append(f"       {hint}")
            # 🟡 中优先级
            if yellows:
                lines.append(f"    🟡 中优先级 ({len(yellows)}条)")
                for a in yellows[:3]:
                    lines.append(f"       {a.get('日期','')} {a.get('标题','')[:60]}")
            # 🟢 summary
            greens = [a for a in announcements if a.get("级别") == "🟢"]
            if greens:
                lines.append(f"    🟢 日常公告 ({len(greens)}条: 关联交易/担保/股东大会等，已过滤)")

        # ---- 催化剂 ----
        catalyst = analysis.get("催化剂", {})
        if catalyst:
            lines.append("")
            lines.append("  [催化剂]")
            pos = catalyst.get("正面", [])
            if pos:
                lines.append("    正面:")
                for p in pos: lines.append(f"      + {p}")
            neg = catalyst.get("负面", [])
            if neg:
                lines.append("    负面:")
                for n in neg: lines.append(f"      - {n}")
            strength = catalyst.get("强度", "")
            if strength: lines.append(f"    催化剂强度: {strength}")

        # ---- 市场预期拆解 ----
        mkt_exp = analysis.get("市场预期拆解", {})
        if mkt_exp:
            lines.append("")
            lines.append("  [市场预期拆解]")
            imp_growth = mkt_exp.get("当前估值隐含的增长率", "")
            if imp_growth: lines.append(f"    估值隐含增长: {imp_growth}")
            concerns = mkt_exp.get("市场主要担忧", [])
            if concerns:
                lines.append(f"    市场担忧: {'; '.join(concerns)}")
            gap = mkt_exp.get("可能的预期差", "")
            if gap: lines.append(f"    预期差: {gap}")
            # A4：预期差三方对照（市场隐含 / 机构一致预期 / 本报告基准情景）
            cmp_lines = mkt_exp.get("预期差对照", [])
            if cmp_lines:
                lines.append("    预期差对照:")
                for c in cmp_lines:
                    lines.append(f"      - {c}")
            concl = mkt_exp.get("预期差结论", "")
            if concl:
                lines.append(f"    预期差结论: {concl}")
        elif analysis.get("市场已定价", ""):
            lines.append("")
            lines.append(f"  [市场已定价] {analysis['市场已定价']}")

        # ---- 证伪条件 ----
        falsify = analysis.get("证伪条件", [])
        if falsify:
            # 过滤空内容或只有"条件X"前缀没有实质描述的项
            clean_falsify = []
            for f in falsify:
                if not isinstance(f, str):
                    continue
                body = f.strip()
                if not body:
                    continue
                # 去掉 "条件X:" / "条件X" 前缀后检查是否还有内容
                stripped = re.sub(r'^条件\d+\s*[:：]?\s*', '', body).strip()
                if stripped and stripped != body.rstrip(':'):
                    clean_falsify.append(body)
            if clean_falsify:
                lines.append("")
                lines.append("  [证伪条件] 以下情况出现则投资逻辑失效:")
                for f in clean_falsify:
                    lines.append(f"    ❌ {f}")

        # ---- 校验Agent ----
        val_notes = analysis.get("校验", [])
        if val_notes:
            lines.append("")
            for v in val_notes:
                lines.append(f"  {v}")

        # ---- 建议 ----
        advice = analysis.get("操作建议", "")
        stop_loss = analysis.get("止损", "")
        exec_state = analysis.get("执行状态", "")
        if advice:
            lines.append("")
            lines.append(f"  [操作建议] {advice}")
        if exec_state:
            lines.append(f"  {exec_state}")
        if stop_loss:
            lines.append(f"  [止损]     {stop_loss}")

        # ---- 框架分歧 ----
        divergence = analysis.get("框架分歧", "")
        if divergence:
            lines.append("")
            lines.append(f"  {divergence}")

        # ---- 结论 ----
        conclusion = analysis.get("结论", "")
        if conclusion:
            lines.append("")
            if isinstance(conclusion, dict):
                lines.append(f"  [综合结论]")
                for key in ["总评", "买入策略", "持有策略", "卖出条件", "预期收益", "持仓周期"]:
                    val = conclusion.get(key, "")
                    if val:
                        lines.append(f"    {key}: {val}")
            else:
                lines.append(f"  [结论] {conclusion}")

        lines.append("")
        lines.append(f"  * 以上基于公开财务数据, 不构成投资建议")
        lines.append(f"=" * 64)

        return "\n".join(lines)

    except Exception as e:
        return f"[Report Error] {e}"


# ============================================================
#  工具8：涨停板池（Phantom Hunter 用）
# ============================================================

def get_limit_up_pool(top_n: int = 30) -> dict:
    """获取今日A股涨停板股票池，按涨幅降序"""
    try:
        headers = {"User-Agent":
                   _USER_AGENT}

        url = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/"
               "json_v2.php/Market_Center.getHQNodeData?"
               "page=1&num=200&sort=changepercent&asc=0&node=hs_a")

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("gbk"))

        results = []
        for s in data:
            name = s.get("name", "")
            code = s.get("code", "")
            try:
                chg = float(s.get("changepercent", 0) or 0)
            except (ValueError, TypeError):
                continue

            # FIX-06：涨跌停判定走交易规则引擎（板块×上市天数×ST），
            # 替代硬编码 9.9/19.9 阈值；北交所30%、主板10%、双创20% 由引擎按代码前缀区分
            if not _rules_is_limit_up(code, name, chg):
                continue
            if "ST" in name or "N" in name:
                continue

            results.append({
                "代码": code,
                "名称": name,
                "涨幅": chg,
                "最新价": s.get("trade", ""),
                "成交量": s.get("volume", ""),
                "成交额": s.get("amount", ""),
                "换手率": s.get("turnoverratio", ""),
                "市盈率": s.get("per", ""),
                "市净率": s.get("pb", ""),
                "市值": s.get("mktcap", ""),
            })

        # 连板检测：对前10只（大概率是连板龙头）调 stock_history 验证
        for r in results[:10]:
            try:
                hist = fetch_stock_history(r["代码"], scale=240, datalen=8)
                streak = 1
                for bar in reversed(hist.get("data", [])[:-1]):
                    o, c = float(bar["open"]), float(bar["close"])
                    if o > 0 and (c - o) / o * 100 >= 9.5:
                        streak += 1
                    else:
                        break
                r["连板"] = f"{streak}连板" if streak >= 2 else "首板"
            except Exception:
                r["连板"] = "首板"
        for r in results[10:]:
            r["连板"] = "未检测"

        return {"涨停板数量": len(results), "列表": results[:top_n]}

    except Exception as e:
        return {"error": f"涨停板查询失败: {str(e)}"}


def _flash_metric(clean: str, keywords: list, exclude_before: str = None):
    """在业绩快报正文中定位关键词，提取金额(统一为亿元)和同比增速(带符号%)。
    兼容两种格式:
      A) 表格: "营业总收入 343,329.01 352,468.92 -2.59"（单位见文首"单位：万元"）
      B) 散文: "营业总收入34.33亿元，同比下降2.59%"
    exclude_before: 关键词前20字符内含该串则跳过（用于区分归母/扣非归母）。
    返回 (amount, yoy)，失败返回 (None, None)。"""
    for kw in keywords:
        start = 0
        while True:
            idx = clean.find(kw, start)
            if idx < 0:
                break
            start = idx + len(kw)
            if exclude_before and exclude_before in clean[max(0, idx - 20):idx]:
                continue
            window = clean[idx + len(kw):idx + len(kw) + 200]
            # A) 表格格式：关键词后紧跟"本期值 上年同期 变动幅度"三个数字
            mt = re.match(r'\s*([\d,]+(?:\.\d+)?)\s+(?:[\d,]+(?:\.\d+)?)\s+(-?[\d.]+)\s*%?', window)
            if mt:
                amount = float(mt.group(1).replace(",", ""))
                if "单位：万" in clean[:2000] or "单位:万" in clean[:2000]:
                    amount /= 10000
                return round(amount, 2), round(float(mt.group(2)), 2)
            # B) 散文格式
            amount = None
            m = re.search(r'([\d,]+(?:\.\d+)?)\s*(亿|万)?\s*元', window)
            if m:
                amount = float(m.group(1).replace(",", ""))
                if m.group(2) == "万":
                    amount /= 10000
                elif m.group(2) is None:
                    amount /= 1e8
                amount = round(amount, 2)
            yoy = None
            m2 = re.search(r'(增长|下降|增加|减少)\s*([\d.]+)\s*%', window)
            if m2:
                sign = -1 if m2.group(1) in ("下降", "减少") else 1
                yoy = round(sign * float(m2.group(2)), 2)
            return amount, yoy
    return None, None


def _extract_flash_report(clean: str, title: str = "") -> dict | None:
    """从业绩快报/业绩预告正文提取结构化数据。返回 None 表示未提取到任何有效数字。

    FIX-09：业绩预告（区间型、非实际值）走 new_listing.parse_performance_forecast，
    返回带 "_预告" 标记的区间数据——不会回灌利润表（merge_flash_into_profit 跳过），
    仅用于公告提示与情景估值联动。"""
    data = {}
    m = re.search(r'(20\d{2}\s*年\s*(?:半年度|一季度|第一季度|前三季度|第三季度|年度)?)', title)
    if m:
        data["报告期"] = m.group(1).replace(" ", "")

    # 业绩预告：区间表述专用解析器（"营收3.83~3.98亿"、"增长25%~30%"、"净利润下滑约30%"）
    if "预告" in title:
        from backend.new_listing import parse_performance_forecast
        fc = parse_performance_forecast(clean, title)
        if not fc:
            return None
        data["_预告"] = True
        data["预告类型"] = fc.get("forecast_type", "不确定")
        if fc.get("rev_min_yi") is not None:
            data["营收区间(亿元)"] = (fc["rev_min_yi"], fc.get("rev_max_yi"))
        if fc.get("rev_growth_min") is not None:
            data["营收同比区间"] = (fc["rev_growth_min"], fc.get("rev_growth_max"))
        if fc.get("np_growth_min") is not None:
            data["归母同比区间"] = (fc["np_growth_min"], fc.get("np_growth_max"))
        if fc.get("np_min_yi") is not None:
            data["归母净利区间(亿元)"] = (fc["np_min_yi"], fc.get("np_max_yi"))
        data["forecast"] = fc
        return data

    rev, rev_yoy = _flash_metric(clean, ["营业总收入", "营业收入"])
    if rev is not None: data["营收(亿元)"] = rev
    if rev_yoy is not None: data["营收同比(%)"] = rev_yoy
    # 表格格式中关键词常被表格换行拆散（"归属于上市公司股东的净…利润"），
    # 用第三字区分：归母="净"，扣非归母="扣"
    gm, gm_yoy = _flash_metric(clean, ["归属于上市公司股东的净"], exclude_before="扣除非经常性损益")
    if gm is None:
        gm, gm_yoy = _flash_metric(clean, ["归属于上市公司股东的净利润"],
                                   exclude_before="扣除非经常性损益")
    if gm is not None: data["归母净利润(亿元)"] = gm
    if gm_yoy is not None: data["归母同比(%)"] = gm_yoy
    kf, kf_yoy = _flash_metric(clean, ["归属于上市公司股东的扣", "扣除非经常性损益"])
    if kf is not None: data["扣非净利润(亿元)"] = kf
    if kf_yoy is not None: data["扣非同比(%)"] = kf_yoy
    return data if len(data) > 1 else None


def _format_flash_hint(flash: dict) -> str:
    """把快报数据格式化成一行提示，供公告列表渲染。"""
    # FIX-09：业绩预告渲染区间（"预告: 归母净利50~55亿(+54%~+69%)，营收3.83~3.98亿"）
    if flash.get("_预告"):
        parts = []
        if flash.get("归母净利区间(亿元)"):
            lo, hi = flash["归母净利区间(亿元)"]
            parts.append(f"归母净利{lo}~{hi}亿" if hi and hi != lo else f"归母净利约{lo}亿")
        if flash.get("归母同比区间"):
            lo, hi = flash["归母同比区间"]
            parts.append(f"({lo*100:+.0f}%~{hi*100:+.0f}%)" if hi and hi != lo else f"({lo*100:+.0f}%)")
        if flash.get("营收区间(亿元)"):
            lo, hi = flash["营收区间(亿元)"]
            parts.append(f"营收{lo}~{hi}亿" if hi and hi != lo else f"营收约{lo}亿")
        if flash.get("营收同比区间"):
            lo, hi = flash["营收同比区间"]
            parts.append(f"营收{lo*100:+.0f}%~{hi*100:+.0f}%" if hi and hi != lo else f"营收{lo*100:+.0f}%")
        period = f"[{flash['报告期']}]" if flash.get("报告期") else ""
        ftype = flash.get("预告类型", "")
        return f"预告{period}{ftype}: " + " ".join(parts) if parts else ""
    parts = []
    for key, yoy_key, label in [("营收(亿元)", "营收同比(%)", "营收"),
                                ("归母净利润(亿元)", "归母同比(%)", "归母净利"),
                                ("扣非净利润(亿元)", "扣非同比(%)", "扣非净利")]:
        if flash.get(key) is not None:
            s = f"{label}{flash[key]}亿"
            if flash.get(yoy_key) is not None:
                s += f"({flash[yoy_key]:+.1f}%)"
            parts.append(s)
    period = f"[{flash['报告期']}]" if flash.get("报告期") else ""
    return f"快报{period}: " + ", ".join(parts) if parts else ""


def fetch_announcement_content(art_code: str) -> str:
    """按 art_code 抓取公告正文（去HTML标签）。主用 np-cnotice 内容接口，失败回退旧接口。
    供 get_recent_announcements 与外部模块（事件分类器/次新股数据包）复用。失败返回 ""。"""
    if not art_code:
        return ""
    for detail_url in (
        f'https://np-cnotice-stock.eastmoney.com/api/content/ann?art_code={art_code}&client_source=web&page_index=1',
        f'https://np-anotice-stock.eastmoney.com/api/security/ann/detail?art_code={art_code}',
    ):
        try:
            req2 = urllib.request.Request(detail_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=8, context=_SSL_CTX) as resp2:
                detail = json.loads(resp2.read().decode("utf-8"))
            text = str(detail.get("data", {}).get("notice_content",
                    detail.get("data", {}).get("content", "")))
            if text:
                return re.sub(r'<[^>]+>', '', text)
        except Exception:
            continue
    return ""


def get_recent_announcements(symbol: str, count: int = 5) -> dict:
    """获取个股最近 N 条公告（东财）。对定增/发行类、业绩快报/预告类公告自动抓取正文提取关键数字。"""
    try:
        url = 'https://np-anotice-stock.eastmoney.com/api/security/ann'
        params = f'sr=-1&page_size={count}&page_index=1&ann_type=A&client_source=web&stock_list={symbol}'
        req = urllib.request.Request(f'{url}?{params}',
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data", {}).get("list", [])

        _DILUTION_KW = ["发行A股", "非公开发行", "定向增发", "募集资金", "发行股份"]
        _FLASH_KW = ["业绩快报", "业绩预告"]
        _IPO_KW = ["上市公告书", "招股说明书"]  # FIX-03/08：IPO 事件参数与次新股数据包需要正文
        results = []
        for it in items[:count]:
            title = it.get("title","").replace("<em>","").replace("</em>","")
            entry = {"日期": it.get("notice_date","")[:10], "标题": title,
                     "art_code": it.get("art_code","")}  # art_code 供外部按需取全文
            # 对定增/发行类、业绩快报/预告类、上市公告书/招股书类公告抓取正文
            if any(kw in title for kw in _DILUTION_KW + _FLASH_KW + _IPO_KW):
                art_code = it.get("art_code","")
                if art_code:
                    try:
                        clean = fetch_announcement_content(art_code)
                        if not clean:
                            raise ValueError("正文为空")
                        if any(kw in title for kw in _DILUTION_KW):
                            # 提取关键数字
                            amounts = re.findall(r'(\d+\.?\d*)\s*[亿万]元', clean)
                            shares = re.findall(r'(\d+\.?\d*)\s*[万]股', clean)
                            if amounts: entry["募资金额"] = amounts[0] + "亿元" if "亿" in clean else amounts[0] + "万元"
                            if shares: entry["发行股数"] = shares[0] + "万股"
                        if any(kw in title for kw in _FLASH_KW):
                            flash = _extract_flash_report(clean, title)
                            if flash:
                                entry["快报数据"] = flash
                                hint = _format_flash_hint(flash)
                                if hint: entry["提示"] = hint
                        entry["摘要"] = clean[:200] + ("..." if len(clean) > 200 else "")
                    except Exception:
                        pass  # 正文抓取失败不影响
            results.append(entry)

        return {"公告数量": len(results), "列表": results}
    except Exception as e:
        return {"error": f"公告查询失败: {str(e)}"}


def get_stock_streak(symbol: str) -> dict:
    """查询单只股票近10日连板情况。返回连板天数、涨停日期列表。"""
    try:
        hist = fetch_stock_history(symbol, scale=240, datalen=12)
        if "error" in hist:
            return {"error": hist["error"]}
        data = hist.get("data", [])
        if not data:
            return {"连板天数": 0, "说明": "无数据"}

        streak = 0
        dates = []
        for bar in reversed(data[:-1]):  # 最新的在最后，跳过今天（已经涨停）
            o, c = float(bar["open"]), float(bar["close"])
            if o > 0 and (c - o) / o * 100 >= 9.5:
                streak += 1
                dates.append(bar["day"])
            else:
                break
        return {
            "代码": symbol,
            "连板天数": streak + 1,  # +1 = 今天
            "涨停日期": [data[-1]["day"]] + dates if data else [],
            "判断": f"{streak+1}连板" if streak >= 1 else "首板"
        }
    except Exception as e:
        return {"error": f"连板查询失败: {str(e)}"}


# ============================================================
#  工具9：概念板块热度扫描（Phantom Hunter 用）
# ============================================================

def get_concept_ranking(top_n: int = 20) -> dict:
    """获取同花顺概念板块名称列表。数据源较慢（~2分钟），有1小时缓存。"""
    from backend import cache
    cached = cache.get("concept_ranking", "all")
    if cached:
        return cached
    try:
        df = ak.stock_board_concept_name_ths()
        concepts = []
        for _, row in df.head(top_n * 3).iterrows():
            name = str(row.get("name", ""))
            code = str(row.get("code", ""))
            if not name or not code:
                continue
            concepts.append({"概念名称": name, "代码": code})
            if len(concepts) >= top_n:
                break
        result = {"概念总数": len(concepts), "列表": concepts}
        cache.set("concept_ranking", "all", result)
        return result
    except Exception as e:
        return {"error": f"概念板块查询失败（同花顺服务器响应慢，请稍后重试）: {str(e)[:100]}"}


# ============================================================
#  确定性评分引擎（替代 LLM 主观打分）
# ============================================================

def _period_rank(period: str) -> int:
    """财报期间序号：一季报<中报/半年度<三季报<年报。用于判断业绩快报是否比结构化数据更新鲜。"""
    if not period:
        return 0
    if "一季报" in period or "一季度" in period or "第一季度" in period:
        return 1
    if "中报" in period or "半年度" in period or "半年" in period:
        return 2
    if "三季报" in period or "三季度" in period or "第三季度" in period:
        return 3
    if "年报" in period or "年度" in period:
        return 4
    return 0


def _flash_period_label(period: str) -> str:
    """快报报告期描述 → 利润表标准期间标签"""
    r = _period_rank(period)
    return {1: "一季报", 2: "半年报", 3: "三季报", 4: "年报"}.get(r, "")


def merge_flash_into_profit(profit_list: list, flash: dict) -> list:
    """把公告快报数据回灌利润表：
    1) 快报期间与利润表最新行一致 → 补齐缺失的扣非/归母/营收字段（datacenter快报行缺扣非）
    2) 快报期间比利润表最新行更新 → 按快报合成新行前置插入
    返回修改后的 profit_list（原地修改）。"""
    if not flash or not isinstance(profit_list, list):
        return profit_list
    # FIX-09：业绩预告是预测值而非实际业绩，禁止回灌利润表（否则污染 TTM/评分）
    if flash.get("_预告"):
        return profit_list
    label = _flash_period_label(flash.get("报告期", ""))
    if not label:
        return profit_list

    def _yi(key):
        v = flash.get(key)
        return v * 1e8 if isinstance(v, (int, float)) else None

    f_rev = _yi("营收(亿元)")
    f_gm = _yi("归母净利润(亿元)")
    f_kf = _yi("扣非净利润(亿元)")

    if profit_list and profit_list[0].get("报告期") == label:
        row = profit_list[0]
        if f_kf is not None and row.get("扣非净利润") is None:
            row["扣非净利润"] = f_kf
        if f_gm is not None and row.get("归母净利润") is None:
            row["归母净利润"] = f_gm
        if f_rev is not None and row.get("营业总收入") is None:
            row["营业总收入"] = f_rev
        return profit_list

    if not profit_list or _period_rank(label) > _period_rank(profit_list[0].get("报告期", "")):
        profit_list.insert(0, {
            "date": "", "报告期": label,
            "营业总收入": f_rev, "归母净利润": f_gm, "扣非净利润": f_kf,
            "_快报源": True,
        })
    return profit_list


def calculate_scores(financial_data: dict) -> dict:
    """纯函数，根据财报数据计算6维评分。数据不足时该维度自动标N/A。"""
    scores = {}

    # Harness: 数据不足拦截
    val_data = financial_data.get("valuation", {}).get("data", [])
    if not val_data and not financial_data.get("profit"):
        return {"盈利能力": {"得分": None, "依据": "数据不足，无法计算"},
                "成长性": {"得分": None, "依据": "数据不足，无法计算"},
                "财务健康": {"得分": None, "依据": "数据不足，无法计算"},
                "估值合理": {"得分": None, "依据": "数据不足，无法计算"},
                "行业前景": {"得分": 5, "依据": "待LLM根据行业信息微调(+-3)"},
                "资金认可": {"得分": 5, "依据": "待LLM根据主力净流入判断(+-5)"}}

    # 从 valuation 数据中提取值——优先年报（避免Q1单季度ROE偏低）
    val_data = financial_data.get("valuation", {}).get("data", [])
    _annual_vals = [v for v in val_data if (v.get("日期") or v.get("date") or "").endswith("-12-31")]
    latest = _annual_vals[0] if _annual_vals else (val_data[0] if val_data else {})
    # 兜底：优先年报，其次遍历所有记录找到第一个有值的
    def _find_val(key):
        # 1) latest (年报优先)
        v = latest.get(key)
        if v is not None:
            try:
                fv = float(v)
                if fv != 0:
                    return fv
            except (ValueError, TypeError):
                pass
        # 2) 从年报列表中找
        for item in (_annual_vals or []):
            if item is latest:
                continue
            v2 = item.get(key)
            if v2 is not None:
                try:
                    fv2 = float(v2)
                    if fv2 != 0:
                        return fv2
                except (ValueError, TypeError):
                    pass
        # 3) 从所有记录找
        for item in (val_data or []):
            if item is latest or item in (_annual_vals or []):
                continue
            v3 = item.get(key)
            if v3 is not None:
                try:
                    fv3 = float(v3)
                    if fv3 != 0:
                        return fv3
                except (ValueError, TypeError):
                    pass
        return 0.0

    # --- 1. 盈利能力 (0-10) ---
    roe = _find_val("ROE(%)")
    gm = _find_val("毛利率(%)")
    nm = _find_val("净利率(%)")

    # 财报期次纪律（用户规则）：最近一份财报为主要参考，较远两份仅用于趋势。
    # 最新期次优先：ROE 用最新期年化值、毛利率/净利率用最新期值；
    # 季节性失真防护（与前瞻PE禁用同源）：Q1净利 < 年报15% → 年化无意义，退回年报口径。
    _period_note = ""
    if val_data and len(val_data) >= 2:
        _q_latest = val_data[0]  # 最新报告期
        _q_date = _q_latest.get("日期") or _q_latest.get("date") or ""
        if not _q_date.endswith("-12-31"):  # 最新期非年报
            try:
                _q_period = _q_latest.get("报告期", "最新期")
                _q_roe = float(_q_latest.get("ROE(%)", 0) or 0)
                _q_gm = float(_q_latest.get("毛利率(%)", 0) or 0)
                _q_nm = float(_q_latest.get("净利率(%)", 0) or 0)
                _ann_factor = {"一季报": 4, "半年报": 2, "三季报": 4/3, "一季度": 4}.get(_q_period, 0)
                if _ann_factor == 0:
                    _mm = _q_date[5:7] if len(_q_date) >= 7 else ""
                    _ann_factor = {"03": 4, "06": 2, "09": 4/3}.get(_mm, 1)
                _seasonal = False
                if _q_period == "一季报":
                    _prof0 = (financial_data.get("profit", []) or [{}])[0]
                    _anns0 = [p for p in (financial_data.get("profit", []) or []) if p.get("报告期") == "年报"]
                    _q_net0 = float(_prof0.get("归母净利润") or _prof0.get("扣非净利润") or 0)
                    _ann_net0 = float(_anns0[0].get("归母净利润") or _anns0[0].get("扣非净利润") or 0) if _anns0 else 0
                    if _ann_net0 > 0 and 0 < _q_net0 < _ann_net0 * 0.15:
                        _seasonal = True
                if _seasonal:
                    _period_note = f"（{_q_period}季节性失真，主参考=年报口径）"
                else:
                    _ann_roe, _ann_gm = roe, gm
                    if _q_roe > 0:
                        roe = _q_roe * _ann_factor
                    if _q_gm > 0:
                        gm = _q_gm
                    if _q_nm > 0:
                        nm = _q_nm
                    _period_note = f"（{_q_period}口径为主，年报ROE{_ann_roe:.1f}%/毛利率{_ann_gm:.1f}%）"
            except (ValueError, TypeError):
                pass  # 季度数据提取失败，保持年报值

    if roe >= 50: pe_score = 10
    elif roe >= 30: pe_score = 9
    elif roe >= 20: pe_score = 7
    elif roe >= 10: pe_score = 5
    elif roe >= 5: pe_score = 3
    else: pe_score = 1

    if gm >= 40: pe_score = min(10, pe_score + 1)
    if nm >= 20: pe_score = min(10, pe_score + 1)
    scores["盈利能力"] = {"得分": pe_score,
                         "依据": f"ROE {roe:.1f}%, 毛利率{gm:.1f}%, 净利率{nm:.1f}%{_period_note}"}

    # --- 2. 成长性 (0-10) —— 年报底色 + 季报势头，扣非净利润优先 ---
    profit_data = financial_data.get("profit", [])
    annuals = [p for p in profit_data if p.get("报告期") == "年报"]
    g_score = 5  # 默认中性
    basis_parts = []

    if len(annuals) >= 2:
        # === 年报YoY：长期趋势（底色）===
        a_cur, a_prev = annuals[0], annuals[1]
        a_rev_cur = float(a_cur.get("营业总收入") or 0)
        a_rev_prev = float(a_prev.get("营业总收入") or 1)
        a_rev_g = (a_rev_cur - a_rev_prev) / a_rev_prev * 100 if a_rev_prev > 0 else 0

        a_net_cur = float(a_cur.get("扣非净利润") or a_cur.get("归母净利润") or 0)
        a_net_prev = float(a_prev.get("扣非净利润") or a_prev.get("归母净利润") or 1)
        a_net_g = (a_net_cur - a_net_prev) / a_net_prev * 100 if a_net_prev > 0 else 0
        metric = "扣非" if a_cur.get("扣非净利润") else "归母"
        basis_parts.append(f"年报:营收{a_rev_g:+.0f}%,{metric}净利润{a_net_g:+.0f}%")

        # 年报得分基线：营收权重40% + 扣非净利润权重60%
        def _growth_score(rev_g, net_g):
            """营收和利润加权打分"""
            def _score(val):
                if val >= 100: return 10
                if val >= 50: return 9
                if val >= 30: return 8
                if val >= 20: return 6
                if val >= 10: return 5
                if val >= 0: return 3
                return 0
            return int(_score(rev_g) * 0.4 + _score(net_g) * 0.6)
        g_score = _growth_score(a_rev_g, a_net_g)

        # === 季报YoY：最新动向（势头）===
        latest = profit_data[0]
        # 业绩快报行（_快报源=True）插入到 position 0 时会破坏季度对比
        # 如果首行是快报源，跳过它用真实季报行
        _is_flash_row0 = latest.get("_快报源", False)
        if _is_flash_row0 and len(profit_data) >= 2:
            latest = profit_data[1]
        latest_period = latest.get("报告期", "")
        # 业绩快报修正：快报覆盖期间比最新结构化季报更新鲜时，以快报为势头信号
        # （否则"Q1扣非-86%但半年度快报扣非+11%"时，评分仍停留在Q1的失真状态）
        flash = financial_data.get("flash") or {}
        flash_period = flash.get("报告期", "")
        use_flash = bool(flash) and _period_rank(flash_period) > _period_rank(latest_period)
        if use_flash:
            f_rev = flash.get("营收同比(%)")
            f_net = flash.get("扣非同比(%)")
            if f_net is None:
                f_net = flash.get("归母同比(%)")
            # 业绩预告提取容错：-100 或 None 意味着 _flash_metric 解析失败（常见于中文标点）
            # 此时忽略 flash 势头信号，回退到季报对比路径
            _flash_unreliable = (f_net is None or f_net == -100 or f_net == -100.0)
            if _flash_unreliable:
                use_flash = False  # 回退到季报势头（下方 elif 分支）
            elif f_net is not None:
                if f_rev is not None:
                    basis_parts.append(f"快报[{flash_period}]:营收{f_rev:+.0f}%,扣非{f_net:+.0f}%")
                else:
                    basis_parts.append(f"快报[{flash_period}]:扣非{f_net:+.0f}%")
                # 势头修正：年报增速 vs 快报增速（替代季报增速）
                if a_net_g > 0 and f_net > 0:
                    if f_net >= a_net_g:
                        g_score = min(10, g_score + 2)  # 加速
                        basis_parts.append("趋势:加速↑(快报)")
                    else:
                        basis_parts.append("趋势:延续→(快报)")
                elif a_net_g > 0 and f_net < 0:
                    g_score = max(0, g_score - 3)
                    basis_parts.append("趋势:拐点恶化⚠️(快报)")
                elif a_net_g < 0 and f_net > 0:
                    g_score = min(10, g_score + 3)  # 年报跌但快报转正→拐点改善
                    basis_parts.append("趋势:拐点改善↑(快报)")
                # 年报/快报均负：不修正
        elif latest_period != "年报":
            q_same = [p for p in profit_data if p.get("报告期") == latest_period]
            if len(q_same) >= 2:
                q_cur, q_prev = q_same[0], q_same[1]
                q_net_cur = float(q_cur.get("扣非净利润") or q_cur.get("归母净利润") or 0)
                q_net_prev = float(q_prev.get("扣非净利润") or q_prev.get("归母净利润") or 1)
                q_net_g = (q_net_cur - q_net_prev) / q_net_prev * 100 if q_net_prev > 0 else 0
                basis_parts.append(f"{latest_period}:{metric}净利润{q_net_g:+.0f}%")

                # 势头修正：年报增速 vs 季报增速 → 四个象限
                if a_net_g > 0 and q_net_g > 0:
                    if q_net_g >= a_net_g:
                        g_score = min(10, g_score + 2)  # 加速
                        basis_parts.append("趋势:加速↑")
                    elif q_net_g >= a_net_g * 0.5:
                        basis_parts.append("趋势:延续→")  # 延续
                    else:
                        g_score = max(0, g_score - 1)  # 放缓
                        basis_parts.append("趋势:放缓↓")
                elif a_net_g > 0 and q_net_g < 0:
                    g_score = max(0, g_score - 3)  # 年报增但季报跌→拐点恶化
                    basis_parts.append("趋势:拐点恶化⚠️")
                elif a_net_g < 0 and q_net_g > 0:
                    g_score = min(10, g_score + 3)  # 年报跌但季报增→拐点改善
                    basis_parts.append("趋势:拐点改善↑")
                # else: both negative, no change
        else:
            basis_parts.append("季报:暂无")
    elif len(profit_data) >= 2:
        # 回退：无年报时用同类型对比
        latest_period = profit_data[0].get("报告期", "")
        same_type = [p for p in profit_data if p.get("报告期") == latest_period]
        if len(same_type) >= 2:
            cur, prev = same_type[0], same_type[1]
            rev_g = (float(cur.get("营业总收入") or 0) - float(prev.get("营业总收入") or 1)) / float(prev.get("营业总收入") or 1) * 100
            net_g = (float(cur.get("扣非净利润") or cur.get("归母净利润") or 0) - float(prev.get("扣非净利润") or prev.get("归母净利润") or 1)) / float(prev.get("扣非净利润") or prev.get("归母净利润") or 1) * 100
            basis_parts.append(f"{latest_period}:营收{rev_g:+.0f}%,净利润{net_g:+.0f}%")
            if rev_g >= 30: g_score = 8
            elif rev_g >= 0: g_score = 3
            else: g_score = 0
    else:
        basis_parts.append("数据不足")

    scores["成长性"] = {"得分": g_score, "依据": "; ".join(basis_parts)}

    # --- 3. 财务健康 (0-10) —— 折旧修正 ---
    debt = _find_val("资产负债率(%)") or 50
    cf_data = financial_data.get("cashflow", [])
    dep = 0  # 折旧摊销（重资产行业现金流修正因子）
    cf_score = 0
    cf_ratio = 0
    cf_period = ""
    op_cf = 0.0
    capex_val = 0.0
    fcf = 0.0
    fcf_ratio = 0.0
    net_profit = 1.0
    _fcf_cfo = 0.0  # FCF 专用 CFO（恒为年报口径；cf_data 为空时保持默认）
    if cf_data:
        # 财报期次纪律（用户规则）：最近一份财报为主要参考——现金流覆盖率用最新期次。
        # 季节性失真防护：Q1净利<年报15% 且 Q1经营现金流为正 → 覆盖率虚高失真，退回年报口径；
        # （Q1现金流为负是真实恶化信号，不属失真，必须保留最新期）。
        # FCF 仍走年报口径（资本开支是年度尺度，单季口径无意义）。
        cf_annual = [c for c in cf_data if c.get("报告期") == "年报"]
        annual_profit = [p for p in profit_data if p.get("报告期") == "年报"]
        _use_latest = True
        _latest_period0 = profit_data[0].get("报告期", "") if profit_data else ""
        if _latest_period0 == "一季报" and profit_data and annual_profit:
            _q_net0 = float(profit_data[0].get("归母净利润") or profit_data[0].get("扣非净利润") or 0)
            _ann_net0 = float(annual_profit[0].get("归母净利润") or annual_profit[0].get("扣非净利润") or 0)
            _q_cfo0 = float(cf_data[0].get("经营现金流净额", 0) or 0)
            if _ann_net0 > 0 and 0 < _q_net0 < _ann_net0 * 0.15 and _q_cfo0 > 0:
                _use_latest = False  # Q1 季节性虚高失真
        if _use_latest:
            cf_latest = cf_data[0]  # 降序排列，[0]最新
            cf_period = cf_latest.get("报告期", "最新期")
            net_profit = float(profit_data[0].get("扣非净利润") or profit_data[0].get("归母净利润") or 1) if profit_data else 1
        elif cf_annual and annual_profit:
            cf_latest = cf_annual[0]
            cf_period = "年报"
            net_profit = float(annual_profit[0].get("扣非净利润") or annual_profit[0].get("归母净利润") or 1)
        else:
            cf_latest = cf_data[0]
            cf_period = cf_latest.get("报告期", "最新期")
            net_profit = float(profit_data[0].get("扣非净利润") or profit_data[0].get("归母净利润") or 1) if profit_data else 1
        op_cf = float(cf_latest.get("经营现金流净额", 0) or 0)
        dep = float(cf_latest.get("折旧摊销", 0) or 0)
        # 扣非净利润（排除资产处置等非经常项目）用于现金流质量判断
        cf_ratio = op_cf / net_profit if net_profit > 0 else 0
        # 折旧修正：重资产行业OFC/NI天然偏高，折旧占比>30%净利润时降低加分门槛
        dep_ratio = dep / net_profit if net_profit > 0 else 0
        if dep_ratio > 0.5:      # 重资产行业：OFC/NI>2才算好
            if cf_ratio >= 2.0: cf_score = 2
            elif cf_ratio >= 1.0: cf_score = 1
        else:                     # 轻资产行业：OFC/NI>0.8即好
            if cf_ratio >= 0.8: cf_score = 2

        # FCF = CFO - CAPEX。重资产行业CFO可能很高但全被CAPEX吃掉
        # FCF 始终用年报口径（资本开支是年度尺度，单季无意义）；覆盖率仍用最新期（期次纪律）
        _fcf_row = cf_annual[0] if cf_annual else cf_latest
        capex_val = float(_fcf_row.get("购建固定资产支付现金", 0) or 0)
        _fcf_cfo = float(_fcf_row.get("经营现金流净额", 0) or 0)
        fcf = _fcf_cfo - capex_val
        if _fcf_cfo > 0 and capex_val > 0:
            fcf_ratio = fcf / _fcf_cfo  # FCF/CFO：现金流中多少是真正自由的
        else:
            fcf_ratio = 0

    if debt < 30: debt_score = 10
    elif debt < 50: debt_score = 7
    elif debt < 60: debt_score = 5
    else: debt_score = 3
    dep_note = f", 折旧/净利润={dep_ratio:.1f}(重资产修正)" if dep > 0 and net_profit > 0 else ""
    # 现金流色彩标签（0-3级，供Reporter强制注入风险段落）
    if cf_ratio >= 0.8:   cf_label, cf_emoji, cf_severity = "优秀", "🟢", 0
    elif cf_ratio >= 0.5: cf_label, cf_emoji, cf_severity = "正常", "🟡", 1
    elif cf_ratio >= 0.3: cf_label, cf_emoji, cf_severity = "警惕", "🟠", 2
    else:                 cf_label, cf_emoji, cf_severity = "警报", "🔴", 3

    # FCF 预警标记（供 _fix_and_decide 强制注入风险段落）
    _fcf_warning = ""
    if _fcf_cfo > 0 and capex_val > 0 and fcf < 0 and fcf_ratio < -0.3:
        _fcf_warning = ("FCF预警: 经营现金流{:.2f}亿但资本开支{:.2f}亿, "
                        "自由现金流≈{:.2f}亿(负!), CFO虽高但被CAPEX吞噬, "
                        "'利润含金量高'仅适用于CFO,不适用于FCF".format(_fcf_cfo/1e8, capex_val/1e8, fcf/1e8))

    scores["财务健康"] = {
        "得分": min(10, debt_score + cf_score),
        "依据": f"资产负债率{debt:.1f}%, 经营现金流/扣非净利润={cf_ratio:.2f}({cf_period}口径){dep_note}",
        "现金流标签": f"{cf_emoji}{cf_label}(覆盖率{cf_ratio:.2f},{cf_period})",
        "现金流严重度": cf_severity,
        "FCF预警": _fcf_warning,
    }

    # --- 4. 估值合理 (0-10) —— 行业PE锚定 ---
    _INDUSTRY_PE = {
        "电力": 15, "银行": 7, "保险": 12, "券商": 18, "地产": 10,
        "钢铁": 12, "化工": 18, "煤炭": 10, "石油": 12, "有色": 22,
        "白酒": 28, "食品": 25, "家电": 15, "汽车": 18, "医药": 30,
        "电子": 30, "半导体": 40, "计算机": 35, "通信": 22, "传媒": 20,
        "新能源": 25, "军工": 40, "机械": 22, "建材": 15, "建筑": 10,
        "交运": 15, "公用事业": 18, "环保": 20, "商贸": 18, "纺织": 18,
        "IT服务": 32, "IT服务Ⅱ": 32, "软件开发": 35, "互联网服务": 30,
        "电池": 25, "乘用车": 18, "元件": 30, "光伏设备": 25,
        "通信设备": 22, "白酒Ⅱ": 28, "银行Ⅱ": 7, "股份制银行": 7,
        "白色家电": 15, "家用电器": 15,
    }
    price_info = financial_data.get("price", {})
    # PE 始终用 TTM 自算，不依赖外部 API（新浪/东财返回的年报 PE 对 V 型反转公司严重失真）
    stock_price = float(price_info.get("price", 0) or 0) if isinstance(price_info, dict) else 0
    if stock_price <= 0:
        stock_price = 0
    # TTM净利润 = 最新年报净利 - 去年同期净利 + 最新同期净利
    # FIX-01：TTM/EPS 计算统一走股本SSOT的 compute_ttm_eps（与 reporter 同一口径）
    ann_net = float(annuals[0].get("归母净利润") or annuals[0].get("扣非净利润") or 0) if annuals else 0
    total_shares = _find_val("总股本")
    eps_ttm, ttm_net = compute_ttm_eps(profit_data, total_shares)
    if ttm_net <= 0:
        ttm_net = ann_net
    if eps_ttm <= 0:
        eps_ttm = _find_val("每股收益")
    pe = stock_price / eps_ttm if stock_price > 0 and eps_ttm > 0 else 0
    pb = float(price_info.get("pb", 0) or 0) if isinstance(price_info, dict) else 0
    # PB未获取时自算 = 股价 / 每股净资产
    if pb <= 0:
        stock_price2 = float(price_info.get("price", 0) or 0) if isinstance(price_info, dict) else 0
        bps = _find_val("每股净资产")
        if stock_price2 > 0 and bps > 0:
            pb = stock_price2 / bps

    industry = financial_data.get("industry", "")
    # 模糊匹配：API返回"白色家电"→映射表"家电"，按最长命中
    ind_pe = _INDUSTRY_PE.get(industry, 0)
    if ind_pe == 0 and industry:
        for k in sorted(_INDUSTRY_PE, key=len, reverse=True):
            if k in industry or industry in k:
                ind_pe = _INDUSTRY_PE[k]
                break
    if ind_pe == 0:
        ind_pe = 18
    if pe <= 0:
        v_score = 5
    else:
        ratio = pe / ind_pe if ind_pe > 0 else 1
        if ratio < 0.6: v_score = 10
        elif ratio < 0.9: v_score = 8
        elif ratio < 1.2: v_score = 6
        elif ratio < 1.6: v_score = 4
        else: v_score = 2
    ind_note = f", 行业PE基准{ind_pe}倍" if industry else ""
    pb_note = f", PB {pb:.1f}倍" if pb > 0 else ""
    scores["估值合理"] = {"得分": v_score, "依据": f"PE {pe:.0f}倍{pb_note}{ind_note}"}

    # --- 5. 行业前景 (0-10, 默认5，LLM可根据行业调整) ---
    scores["行业前景"] = {"得分": 5, "依据": "待LLM根据行业信息微调(+-3)"}

    # --- 6. 资金认可 (0-10) ---
    fund_data = financial_data.get("fund_flow", {})
    if "error" in str(fund_data):
        scores["资金认可"] = {"得分": None, "依据": "数据缺失"}
    else:
        scores["资金认可"] = {"得分": 5, "依据": "待LLM根据主力净流入判断(+-5)"}

    return scores


# ============================================================
#  工具10：龙虎榜（Phantom Hunter 用）
# ============================================================

# 知名游资及关联营业部（持续更新）
YOUZI_DB = {
    "炒股养家": {"席位": ["华鑫证券上海分公司","华鑫证券上海宛平南路","华鑫证券上海松江"],"风格":"格局锁仓,不轻易卖,偏好科技"},
    "方新侠":   {"席位": ["中信证券上海分公司","中信证券上海溧阳路"],"风格":"打板猛,次日高开出货,一日游为主"},
    "上塘路":   {"席位": ["中信证券杭州上塘路","中信证券杭州延安路"],"风格":"跟风助攻,快进快出"},
    "作手新一": {"席位": ["国泰君安南京太平南路","国泰君安上海分公司"],"风格":"题材挖掘,持股周期适中"},
    "赵老哥":   {"席位": ["中国银河证券上海杨浦区","中国银河证券北京"],"风格":"消息驱动,打板不恋战"},
    "小鳄鱼":   {"席位": ["东方证券上海浦东新区","东方证券上海静安区"],"风格":"趋势接力,偏好新能源"},
    "章盟主":   {"席位": ["国泰君安上海分公司","海通证券上海"],"风格":"锁仓+低吸,偏好白马"},
}

def identify_youzi(branch_name: str) -> list:
    """识别营业部对应哪些游资"""
    matched = []
    for name, info in YOUZI_DB.items():
        for seat in info["席位"]:
            if seat[:6] in branch_name or branch_name[:4] in seat:
                matched.append({"游资": name, "风格": info["风格"]})
                break
    return matched


def get_dragon_tiger_list(date: str = "") -> dict:
    """获取今日龙虎榜数据，识别游资席位（5分钟缓存）"""
    from backend import cache
    cached = cache.get("dragon_tiger_list", date or "today")
    if cached:
        return cached
    try:
        df = ak.stock_lhb_ggtj_sina()
        # 取最近30条
        df = df.head(30)

        results = []
        for _, row in df.iterrows():
            results.append({
                "代码": row["股票代码"],
                "名称": row["股票名称"],
                "上榜次数": row.get("上榜次数", ""),
                "买入总额": str(row.get("累积买入额", "")),
                "卖出总额": str(row.get("累积卖出额", "")),
                "净买入额": str(row.get("净买入额", "")),
            })

        result = {"龙虎榜数量": len(results), "列表": results}
        cache.set("dragon_tiger_list", date or "today", result)
        return result

    except Exception as e:
        return {"error": f"龙虎榜查询失败: {str(e)}"}


def get_dragon_tiger_detail(symbol: str) -> dict:
    """获取个股龙虎榜统计（上榜次数/买卖总额/机构追踪）"""
    try:
        # 从龙虎榜个股统计中查
        df = ak.stock_lhb_ggtj_sina()
        stock_row = df[df["股票代码"] == symbol]

        if stock_row.empty:
            # 尝试从机构追踪中查
            df2 = ak.stock_lhb_jgzz_sina()
            stock_row2 = df2[df2["股票代码"] == symbol]
            if stock_row2.empty:
                return {"error": f"{symbol} 近期未上龙虎榜"}

            r = stock_row2.iloc[0]
            return {
                "代码": symbol,
                "名称": str(r.get("股票名称", "")),
                "累积买入额": str(r.get("累积买入额", "")),
                "累积卖出额": str(r.get("累积卖出额", "")),
                "净买入额": str(r.get("净买入额", "")),
                "买入次数": str(r.get("买入次数", "")),
                "卖出次数": str(r.get("卖出次数", "")),
            }

        r = stock_row.iloc[0]
        buy_total = str(r.get("累积买入额", ""))
        sell_total = str(r.get("累积卖出额", ""))
        net = str(r.get("净买入额", ""))

        # 结合机构席位数据
        try:
            detail = ak.stock_lhb_jgmx_sina()
            stock_detail = detail[detail["股票代码"] == symbol]
            buy_amt = stock_detail["买方席位买入额"].sum() if not stock_detail.empty else 0
            sell_amt = stock_detail["卖方席位卖出额"].sum() if not stock_detail.empty else 0
        except Exception:
            buy_amt, sell_amt = 0, 0

        return {
            "代码": symbol,
            "名称": str(r.get("股票名称", "")),
            "上榜次数": str(r.get("上榜次数", "")),
            "累积买入额": buy_total,
            "累积卖出额": sell_total,
            "净买入额": net,
            "买方席位合计": str(buy_amt),
            "卖方席位合计": str(sell_amt),
            "资金方向": "净流入" if float(net or 0) > 0 else "净流出",
        }

    except Exception as e:
        return {"error": f"个股龙虎榜查询失败: {str(e)}"}


# ============================================================
#  工具11：板块资金流向（Phantom Hunter / 市场感知用）
# ============================================================

def get_sector_fund_flow(top_n: int = 50, date: str = "", fund_type: str = "total") -> dict:
    """获取全行业板块资金流向排名（同花顺）。
    fund_type: 'total'=全市场资金(主力+散户) / 'main'=主力资金(超大单+大单)"""
    try:
        import requests
        from bs4 import BeautifulSoup as bs

        headers = _get_ths_headers()
        # URL 路径: hyzjl = 全市场 / hyzjl/field/zljlr = 主力净流入
        path = "hyzjl" if fund_type == "total" else "hyzjl/field/zljlr"
        label = "全市场资金" if fund_type == "total" else "主力资金"
        all_sectors = []
        for page in range(1, 5):
            url = (f"http://data.10jqka.com.cn/funds/{path}/"
                   f"order/desc/page/{page}/ajax/1/free/1/")
            resp = requests.get(url, headers=headers, timeout=10)
            soup = bs(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                break

            for row in table.find_all("tr")[1:]:
                cols = [td.text.strip() for td in row.find_all("td")]
                if len(cols) < 6:
                    continue
                # cols: [排名, 板块名称, 板块指数, 涨跌幅, 流入(亿), 流出(亿)]
                try:
                    inflow = float(cols[4])
                    outflow = float(cols[5])
                    net = inflow - outflow
                except (ValueError, IndexError):
                    net = 0
                    inflow = outflow = 0

                all_sectors.append({
                    "板块": cols[1],
                    "涨跌幅": cols[3],
                    "流入(亿)": round(inflow, 2),
                    "流出(亿)": round(outflow, 2),
                    "净额(亿)": round(net, 2),
                    "方向": "流入" if net >= 0 else "流出",
                })

        all_sectors.sort(key=lambda x: x["净额(亿)"], reverse=True)
        from datetime import datetime
        return {"板块数量": len(all_sectors), "列表": all_sectors[:top_n], "资金类型": label,
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M"), "数据日期": datetime.now().strftime("%Y-%m-%d")}

    except Exception as e:
        return {"error": f"板块资金流查询失败: {str(e)}"}


# ============================================================
#  板块动量评分（中线动量聚焦）
# ============================================================

def get_sector_momentum(top_n: int = 15) -> dict:
    """中线动量聚焦：综合资金流向+涨跌幅+市场情绪，计算板块动量分数。
    不预测未来，追踪当前主力共识最强的板块——'趋势中继'而非'底部反转'。

    返回: {板块, 动量分数, 温度计, 净流入, 涨跌幅, 总成交, 逻辑简述}
    """
    try:
        from datetime import datetime

        # 1. 获取全市场资金流数据
        sf = get_sector_fund_flow(100, fund_type="total")
        sectors = sf.get("列表", [])
        if not sectors:
            return {"error": "板块资金流数据不可用", "列表": []}

        # 2. 获取市场宽度作为情绪背景
        breadth = get_market_breadth()
        up_ratio = float(str(breadth.get("上涨比例", "50")).replace("%", ""))
        # 市场情绪因子：上涨>60%偏乐观(×1.1), <30%偏悲观(×0.9)
        mood_mult = 1.1 if up_ratio > 60 else (0.9 if up_ratio < 30 else 1.0)

        # 3. 归一化辅助（确保所有值都是 float）
        def _norm(values):
            vals = [float(v) for v in values]
            mn, mx = min(vals), max(vals)
            if mx == mn: return [50] * len(vals)
            return [(v - mn) / (mx - mn) * 100 for v in vals]

        nets_raw = [float(s["净额(亿)"]) for s in sectors]
        changes_raw = [float(str(s.get("涨跌幅", "0%")).replace("%", "").replace("+", "")) for s in sectors]
        totals = [float(s["流入(亿)"]) + float(s["流出(亿)"]) for s in sectors]
        # 资金强度：仅统计净流入板块，流出板块资金分为0
        pos_nets = [max(0, n) for n in nets_raw]  # 流出→0
        intensities = [abs(n) / t if t > 0 else 0 for n, t in zip(nets_raw, totals)]

        norm_nets = _norm(pos_nets)
        norm_changes = _norm([max(0, float(c)) for c in changes_raw])  # 下跌板块涨幅分为0
        norm_intensity = _norm(intensities)

        # 4. 复合评分：资金强度40% + 涨跌幅35% + 集中度25%。流出/下跌板块天然低分
        results = []
        for i, s in enumerate(sectors):
            net = nets_raw[i]
            score = (norm_nets[i] * 0.40 + norm_changes[i] * 0.35 + norm_intensity[i] * 0.25)
            score = round(score * mood_mult, 1)

            # 温度计
            if score >= 80:   temp = "🔥高潮期"
            elif score >= 60: temp = "⚡加速期"
            elif score >= 40: temp = "🌡️升温中"
            else:             temp = "❄️观望"

            # 逻辑简述
            direction = "净流入" if net > 0 else "净流出"
            logic = f"{direction}{abs(net):.1f}亿, 涨跌幅{float(changes_raw[i]):+.2f}%"
            if intensities[i] > 0.5:
                logic += ", 资金高度集中"

            results.append({
                "板块": s["板块"], "动量分数": score, "温度计": temp,
                "净流入(亿)": round(net, 1), "涨跌幅": f"{changes_raw[i]:+.2f}%",
                "总成交(亿)": round(totals[i], 1), "逻辑": logic,
            })

        # 按动量分数排序，仅保留净流入板块（主力真正在买的）
        results.sort(key=lambda x: x["动量分数"], reverse=True)
        inflow_only = [r for r in results if r["净流入(亿)"] > 0]
        return {"板块数量": len(inflow_only), "列表": inflow_only[:top_n],
                "市场情绪": f"{'偏乐观' if mood_mult > 1 else '偏悲观' if mood_mult < 1 else '中性'}(上涨比{up_ratio:.0f}%)",
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M")}

    except Exception as e:
        return {"error": f"动量评分计算失败: {str(e)}", "列表": []}


# ============================================================
#  工具12：日内分时数据
# ============================================================

def get_intraday(symbol: str) -> dict:
    """获取当日5分钟K线（48根/天），用于画分时图"""
    # Harness: 交易时段判断
    from datetime import datetime
    now = datetime.now()
    if now.weekday() >= 5:
        return {"info": "非交易日（周末），无分时数据"}
    t = now.hour * 60 + now.minute
    if t < 9 * 60 + 15 or t > 15 * 60 + 5:
        return {"info": f"非交易时段（当前{now.strftime('%H:%M')}），无分时数据。交易时间: 9:15-15:05"}

    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        return {"error": f"无法识别的股票代码: {symbol}"}

    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={full_code}&scale=5&ma=no&datalen=240"
    )

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", _USER_AGENT)
        req.add_header("Referer", "https://finance.sina.com.cn")

        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        bars = [b for b in data if b["day"].startswith(today)]

        if not bars:
            return {"error": f"今日({today})暂无分时数据（可能未开盘或已收盘数据延迟）"}

        return {
            "symbol": symbol,
            "date": today,
            "bars": [{"time": b["day"][-8:], "open": float(b["open"]),
                      "high": float(b["high"]), "low": float(b["low"]),
                      "close": float(b["close"]), "volume": int(b["volume"])}
                     for b in bars],
            "count": len(bars),
        }

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"分时查询异常: {str(e)}"}


# ============================================================
#  工具13：市场涨跌全景（Market Breadth）
# ============================================================

def get_market_breadth() -> dict:
    """获取全A股/行业/概念的涨跌家数统计"""
    # 先查缓存
    from backend import cache
    cached = cache.get("market_breadth", "all")
    if cached:
        return cached
    try:
        headers = {"User-Agent": _USER_AGENT}
        ctx = _SSL_CTX

        # 全A股: 从新浪拉2页数据统计涨跌
        up_count = down_count = flat_count = 0
        total = 0
        for page in range(1, 3):
            url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/"
                   f"json_v2.php/Market_Center.getHQNodeData?"
                   f"page={page}&num=100&sort=code&asc=1&node=hs_a")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode("gbk"))
            if not data:
                break
            for s in data:
                try:
                    chg = float(s.get("changepercent", 0) or 0)
                except (ValueError, TypeError):
                    continue
                if chg > 0:
                    up_count += 1
                elif chg < 0:
                    down_count += 1
                else:
                    flat_count += 1
                total += 1

        if total == 0:
            return {"error": "未获取到市场数据"}

        result = {
            "全A": {"上涨": up_count, "下跌": down_count, "平盘": flat_count, "总计": total},
            "上涨比例": f"{up_count/total*100:.1f}%",
        }
        cache.set("market_breadth", "all", result)
        return result

    except Exception as e:
        return {"error": f"市场全景查询失败: {str(e)}"}


def market_sentiment_score(symbol: str,
                           price_data: dict = None,
                           market_breadth_data: dict = None,
                           limit_up_data: dict = None,
                           dragon_tiger_data: dict = None) -> dict:
    """
    综合市场情绪评分，作为投资决策的参考项（不参与估值计算）。

    评分维度：
    - 市场广度（40%）：全市场上涨比例
    - 个股涨跌（40%）：目标股票相对昨收涨跌幅
    - 短线热度（20%）：是否涨停/龙虎榜

    返回 -1 ~ +1 的综合情绪得分，及分项说明与操作建议。
    """
    from backend import cache

    # 缓存最终得分
    cached = cache.get("market_sentiment", symbol)
    if cached:
        return cached

    # 1. 市场广度得分
    breadth_score = 0.0
    breadth_note = "数据缺失"
    if market_breadth_data is None:
        market_breadth_data = get_market_breadth()
    if isinstance(market_breadth_data, dict) and "error" not in market_breadth_data:
        up_ratio_str = market_breadth_data.get("上涨比例", "0%")
        breadth_note = up_ratio_str
        try:
            up_ratio = float(up_ratio_str.replace("%", "")) / 100
            if up_ratio >= 0.7:
                breadth_score = 1.0
            elif up_ratio >= 0.55:
                breadth_score = 0.5
            elif up_ratio <= 0.3:
                breadth_score = -1.0
            elif up_ratio <= 0.45:
                breadth_score = -0.5
            else:
                breadth_score = 0.0
        except (ValueError, TypeError):
            pass

    # 2. 个股涨跌幅得分
    stock_score = 0.0
    stock_note = "数据缺失"
    if price_data and isinstance(price_data, dict) and "error" not in price_data:
        try:
            price = float(price_data.get("price", 0) or 0)
            yesterday = float(price_data.get("yesterday_close", 0) or 0)
            stock_note = f"当前{price} / 昨收{yesterday}"
            if yesterday > 0:
                chg = (price - yesterday) / yesterday
                if chg >= 0.05:
                    stock_score = 1.0
                elif chg >= 0.02:
                    stock_score = 0.5
                elif chg <= -0.05:
                    stock_score = -1.0
                elif chg <= -0.02:
                    stock_score = -0.5
                else:
                    stock_score = 0.0
        except (ValueError, TypeError):
            pass

    # 3. 短线热度
    heat_score = 0.0
    heat_notes = []
    if limit_up_data is None:
        limit_up_data = get_limit_up_pool(top_n=30)
    if isinstance(limit_up_data, dict) and "error" not in limit_up_data and "列表" in limit_up_data:
        for r in limit_up_data.get("列表", []):
            if r.get("代码") == symbol:
                heat_score = 1.0
                heat_notes.append("当日涨停")
                break
    if not heat_notes:
        if dragon_tiger_data is None:
            dragon_tiger_data = get_dragon_tiger_list()
        if isinstance(dragon_tiger_data, dict) and "error" not in dragon_tiger_data and "列表" in dragon_tiger_data:
            for r in dragon_tiger_data.get("列表", []):
                if r.get("代码") == symbol:
                    heat_score = 0.8
                    heat_notes.append("当日龙虎榜")
                    break

    # 综合得分
    overall = breadth_score * 0.4 + stock_score * 0.4 + heat_score * 0.2
    overall = round(max(-1.0, min(1.0, overall)), 2)

    # 情绪标签
    if overall >= 0.8:
        label = "极度乐观"
    elif overall >= 0.4:
        label = "偏热"
    elif overall >= 0.1:
        label = "温和偏暖"
    elif overall <= -0.8:
        label = "极度悲观"
    elif overall <= -0.4:
        label = "偏冷"
    elif overall <= -0.1:
        label = "温和偏冷"
    else:
        label = "中性"

    def _suggestion(score: float) -> str:
        if score >= 0.8:
            return "市场情绪极度乐观，注意追高风险；若基本面支持，可持仓但避免新开重仓。"
        elif score >= 0.4:
            return "市场情绪偏暖，基本面买点附近可考虑分批建仓。"
        elif score >= 0.1:
            return "市场情绪温和偏暖，可按估值锚点正常执行。"
        elif score <= -0.8:
            return "市场情绪极度悲观，若基本面未恶化，可能是左侧布局窗口，但需控制仓位。"
        elif score <= -0.4:
            return "市场情绪偏冷，建议耐心观察，等待情绪企稳或基本面催化。"
        elif score <= -0.1:
            return "市场情绪温和偏冷，不追涨，等待更明确信号。"
        else:
            return "市场情绪中性，按基本面估值锚点执行，不因为情绪改变决策。"

    result = {
        "综合情绪得分": overall,
        "情绪标签": label,
        "市场广度": {"上涨比例": breadth_note, "得分": breadth_score},
        "个股涨跌": {"说明": stock_note, "得分": stock_score},
        "短线热度": {"备注": "、".join(heat_notes) if heat_notes else "无", "得分": heat_score},
        "对操作建议": _suggestion(overall),
    }
    cache.set("market_sentiment", symbol, result)
    return result
