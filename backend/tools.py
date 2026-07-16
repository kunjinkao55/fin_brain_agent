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
    """查询A股实时价格（数据源可配置）"""
    if _should_skip("stock_price"):
        return {"error": "stock_price 已连续失败3次，本轮跳过（熔断）"}
    dup = _dedup_check(symbol, "stock_price")
    if dup: return dup
    if _DATA_MODE == "remote":
        result = _remote_fetch("api/data/stock_price", {"symbol": symbol})
        _dedup_record(symbol, "stock_price", result)
        return result
    src = _get_source("stock_price")
    if src not in ("sina", "akshare"):
        return _source_error(src, "仅支持 sina/akshare")

    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        return {"error": f"无法识别的股票代码: {symbol}"}

    url = f"https://hq.sinajs.cn/list={full_code}"

    # 先查缓存
    from backend import cache
    cached = cache.get("stock_price", symbol)
    if cached:
        return cached

    try:
        req = urllib.request.Request(url)
        req.add_header("Referer", "https://finance.sina.com.cn")
        req.add_header("User-Agent",
                       _USER_AGENT)

        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            text = resp.read().decode("gbk")

        match = re.search(r'"([^"]+)"', text)
        if not match:
            return {"error": f"未找到股票数据: {text[:100]}"}

        fields = match.group(1).split(",")
        if len(fields) < 32:
            return {"error": f"返回数据字段不足: {len(fields)}"}

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
        return result

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}


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
        cache.set("financial_statements", symbol, result)
        return result

    except Exception as e:
        return {"error": f"获取财报失败: {str(e)}"}


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
        return {"error": f"获取估值失败: {str(e)}"}


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
        return {"error": f"获取行业信息失败: {str(e)}"}


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

        return {"error": f"未在资金流排名前5页找到 {symbol}"}

    except Exception as e:
        return {"error": f"资金流查询失败: {str(e)}"}


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
            margin = rating.get("实际安全边际", rating.get("安全边际要求", rating.get("安全边际", "?")))
            buy_zone = rating.get("买入区间", "?")
            gap = rating.get("估值差距", "")
            weighted = rating.get("加权总分", "")
            conf = rating.get("置信度", "")
            level_icon = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(level, "⚪")
            lines.append(f"  [投资决策] {level_icon} {level}  合理价值: {fair}  安全边际: {margin}")
            if gap: lines.append(f"    估值差距: {gap}")
            lines.append(f"    安全买入价: {buy_zone}  (需{rating.get('安全边际要求','?')}安全边际)")
            if weighted: lines.append(f"    加权总分: {weighted}/100  置信度: {conf}")
            lines.append(f"    * 估值基于财报数据计算，非实时定价。不构成买卖建议。")

        # 估值方法
        val_method = analysis.get("估值方法", "")
        if val_method:
            lines.append("")
            lines.append(f"  [估值方法] {val_method}")

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
                s_val = info.get("得分")
                reason = info.get("依据", "")
                is_aggregate = dim in _AGGREGATE_KEYS or (isinstance(s_val, (int, float)) and s_val > 10)
                if s_val is None:
                    # 数据缺失，不参与计分
                    lines.append(f"  {dim:<8}  {'N/A':>4}  {'-':<6}  {reason}")
                else:
                    g = grade(s_val) if not is_aggregate else "-"
                    lines.append(f"  {dim:<8}  {s_val:>4}  {g:<6}  {reason}")
                    if not is_aggregate:
                        total += s_val
                        max_total += 10

            lines.append(f"  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*44}")
            if max_total > 0:
                lines.append(f"  {'合计':<8}  {total:>4}/{max_total}   {'':<6}  " +
                             f"{'S:>=9 A:>=7 B:>=5 C:<5'}")

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
        lines.append(f"    * 基于最新财报数据计算，非实时行情。日内股价波动会导致PE/PB/市值变化。请以交易软件实时数据为准。")

        # ---- 机构共识 (Web Search) ----
        consensus = analysis.get("机构共识", {})
        if consensus:
            tp = consensus.get("目标价", {})
            net = consensus.get("净利润预测", {})
            ratings = consensus.get("评级分布", {})
            if tp.get("平均"):
                lines.append("")
                lines.append("  [机构共识]")
                tp_str = f"目标价: 平均{tp['平均']}元"
                if tp.get("最高"): tp_str += f" 最高{tp['最高']}元"
                if tp.get("最低"): tp_str += f" 最低{tp['最低']}元"
                if tp.get("机构数"): tp_str += f" ({tp['机构数']}家机构)"
                lines.append(f"    {tp_str}")
            if net.get("2026"):
                lines.append(f"    2026净利润一致预期: {net['2026']}亿" +
                            (f" ({net['机构数']}家机构)" if net.get("机构数") else ""))
            if any(ratings.get(k) for k in ["买入","增持","中性","减持"]):
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
                    prob = s.get("概率", "?")
                    assumption = s.get("假设", "")
                    lines.append(f"    {info} {scenario}({prob}): {price} — {assumption}")

        # ---- 对比分析 ----
        compare = analysis.get("对比分析", {})
        if compare:
            lines.append(_format_compare_section(compare))

        # ---- 观察指标 ----
        watch = analysis.get("观察指标", [])
        if watch:
            lines.append("")
            lines.append("  [观察指标]")
            for w in watch:
                lines.append(f"    - {w}")

        # ---- 定增信息（代码修正痕迹，供审计参考）----
        zj_info = analysis.get("定增信息", {})
        if isinstance(zj_info, dict) and zj_info.get("摊薄调整系数"):
            lines.append("")
            lines.append(f"  [定增信息] 代码已自动修正: 发行{zj_info.get('发行股数','?')}, "
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
        elif analysis.get("市场已定价", ""):
            lines.append("")
            lines.append(f"  [市场已定价] {analysis['市场已定价']}")

        # ---- 证伪条件 ----
        falsify = analysis.get("证伪条件", [])
        if falsify:
            lines.append("")
            lines.append("  [证伪条件] 以下情况出现则投资逻辑失效:")
            for f in falsify:
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

            # 涨停板阈值：主板≥9.9%，双创≥19.9%，排除只触及未封死的
            if chg >= 19.9:
                pass  # 20cm涨停
            elif 9.9 <= chg < 10.5:
                pass  # 10cm涨停(含少量溢价)
            else:
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


def get_recent_announcements(symbol: str, count: int = 5) -> dict:
    """获取个股最近 N 条公告（东财）。对定增/发行类公告自动抓取摘要中的关键数字。"""
    try:
        url = 'https://np-anotice-stock.eastmoney.com/api/security/ann'
        params = f'sr=-1&page_size={count}&page_index=1&ann_type=A&client_source=web&stock_list={symbol}'
        req = urllib.request.Request(f'{url}?{params}',
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data", {}).get("list", [])

        results = []
        for it in items[:count]:
            title = it.get("title","").replace("<em>","").replace("</em>","")
            entry = {"日期": it.get("notice_date","")[:10], "标题": title}
            # 对定增/发行类公告抓取摘要
            if any(kw in title for kw in ["发行A股","非公开发行","定向增发","募集资金","发行股份"]):
                art_code = it.get("art_code","")
                if art_code:
                    try:
                        detail_url = f'https://np-anotice-stock.eastmoney.com/api/security/ann/detail?art_code={art_code}'
                        req2 = urllib.request.Request(detail_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req2, timeout=8, context=_SSL_CTX) as resp2:
                            detail = json.loads(resp2.read().decode("utf-8"))
                        text = str(detail.get("data", {}).get("notice_content",
                                detail.get("data", {}).get("content", "")))
                        # 提取关键数字
                        amounts = re.findall(r'(\d+\.?\d*)\s*[亿万]元', text)
                        shares = re.findall(r'(\d+\.?\d*)\s*[万]股', text)
                        if amounts: entry["募资金额"] = amounts[0] + "亿元" if "亿" in text else amounts[0] + "万元"
                        if shares: entry["发行股数"] = shares[0] + "万股"
                        # 取正文开头作为摘要（跳过HTML标签）
                        clean = re.sub(r'<[^>]+>', '', text)
                        entry["摘要"] = clean[:200] + ("..." if len(clean) > 200 else "")
                    except Exception:
                        pass  # 摘要抓取失败不影响
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
    """获取同花顺概念板块，按当日涨幅排序"""
    try:
        # 获取所有概念板块
        df = ak.stock_board_concept_name_ths()
        concepts = []
        for _, row in df.iterrows():
            try:
                detail = ak.stock_board_concept_cons_ths(symbol=row["name"])
                # 提取概念板块当日的涨跌幅等指标
                concepts.append({
                    "概念名称": row["name"],
                    "代码": row["code"],
                })
            except Exception:
                continue

        if not concepts:
            return {"error": "概念板块数据获取失败"}

        return {"概念总数": len(concepts), "列表": concepts[:top_n]}

    except Exception as e:
        return {"error": f"概念板块查询失败: {str(e)}"}


# ============================================================
#  确定性评分引擎（替代 LLM 主观打分）
# ============================================================

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

    if roe >= 50: pe_score = 10
    elif roe >= 30: pe_score = 9
    elif roe >= 20: pe_score = 7
    elif roe >= 10: pe_score = 5
    elif roe >= 5: pe_score = 3
    else: pe_score = 1

    if gm >= 40: pe_score = min(10, pe_score + 1)
    if nm >= 20: pe_score = min(10, pe_score + 1)
    scores["盈利能力"] = {"得分": pe_score, "依据": f"ROE {roe}%, 毛利率{gm}%, 净利率{nm}%"}

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
        latest_period = latest.get("报告期", "")
        if latest_period != "年报":
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
    if cf_data:
        cf_latest = cf_data[0]  # 降序排列，[0]最新
        op_cf = float(cf_latest.get("经营现金流净额", 0) or 0)
        dep = float(cf_latest.get("折旧摊销", 0) or 0)
        net_profit = float(profit_data[0].get("扣非净利润") or profit_data[0].get("归母净利润") or 1) if profit_data else 1
        # 扣非净利润（排除资产处置等非经常项目）用于现金流质量判断
        cf_ratio = op_cf / net_profit if net_profit > 0 else 0
        # 折旧修正：重资产行业OFC/NI天然偏高，折旧占比>30%净利润时降低加分门槛
        dep_ratio = dep / net_profit if net_profit > 0 else 0
        if dep_ratio > 0.5:      # 重资产行业：OFC/NI>2才算好
            if cf_ratio >= 2.0: cf_score = 2
            elif cf_ratio >= 1.0: cf_score = 1
        else:                     # 轻资产行业：OFC/NI>0.8即好
            if cf_ratio >= 0.8: cf_score = 2

    if debt < 30: debt_score = 10
    elif debt < 50: debt_score = 7
    elif debt < 60: debt_score = 5
    else: debt_score = 3
    dep_note = f", 折旧/净利润={dep_ratio:.1f}(重资产修正)" if dep > 0 and net_profit > 0 else ""
    scores["财务健康"] = {"得分": min(10, debt_score + cf_score), "依据": f"资产负债率{debt}%, 经营现金流/扣非净利润={cf_ratio:.2f}{dep_note}"}

    # --- 4. 估值合理 (0-10) —— 行业PE锚定 ---
    _INDUSTRY_PE = {
        "电力": 15, "银行": 7, "保险": 12, "券商": 18, "地产": 10,
        "钢铁": 12, "化工": 18, "煤炭": 10, "石油": 12, "有色": 22,
        "白酒": 28, "食品": 25, "家电": 15, "汽车": 18, "医药": 30,
        "电子": 30, "半导体": 40, "计算机": 35, "通信": 22, "传媒": 20,
        "新能源": 25, "军工": 40, "机械": 22, "建材": 15, "建筑": 10,
        "交运": 15, "公用事业": 18, "环保": 20, "商贸": 18, "纺织": 18,
    }
    price_info = financial_data.get("price", {})
    pe = float(price_info.get("per", 0) or 0) if isinstance(price_info, dict) else 0
    # PE未从外部获取时，自算 = 股价 / TTM_EPS（滚动12个月）
    if pe <= 0:
        stock_price = float(price_info.get("price", 0) or 0) if isinstance(price_info, dict) else 0
        # TTM净利润: 最新年报净利 - 去年Q1净利 + 最新Q1净利
        q1s = [p for p in profit_data if p.get("报告期") == "一季报"]
        ann_net = float(annuals[0].get("归母净利润") or annuals[0].get("扣非净利润") or 0) if annuals else 0
        if len(q1s) >= 2:
            ttm_net = ann_net + float(q1s[0].get("归母净利润") or q1s[0].get("扣非净利润") or 0) \
                               - float(q1s[1].get("归母净利润") or q1s[1].get("扣非净利润") or 0)
        else:
            ttm_net = ann_net
        total_shares = _find_val("总股本")
        eps_ttm = ttm_net / total_shares if total_shares > 0 and ttm_net > 0 else 0
        if eps_ttm <= 0:
            eps_ttm = _find_val("每股收益")  # 回退到年报EPS
        if stock_price > 0 and eps_ttm > 0:
            pe = stock_price / eps_ttm
    pb = float(price_info.get("pb", 0) or 0) if isinstance(price_info, dict) else 0
    # PB未获取时自算 = 股价 / 每股净资产
    if pb <= 0:
        stock_price2 = float(price_info.get("price", 0) or 0) if isinstance(price_info, dict) else 0
        bps = _find_val("每股净资产")
        if stock_price2 > 0 and bps > 0:
            pb = stock_price2 / bps

    industry = financial_data.get("industry", "")
    ind_pe = _INDUSTRY_PE.get(industry, 18)
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
    """获取今日龙虎榜数据，识别游资席位"""
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

        return {"龙虎榜数量": len(results), "列表": results}

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
        return {"板块数量": len(all_sectors), "列表": all_sectors[:top_n], "资金类型": label}

    except Exception as e:
        return {"error": f"板块资金流查询失败: {str(e)}"}


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

        return {
            "全A": {"上涨": up_count, "下跌": down_count, "平盘": flat_count, "总计": total},
            "上涨比例": f"{up_count/total*100:.1f}%",
        }

    except Exception as e:
        return {"error": f"市场全景查询失败: {str(e)}"}
