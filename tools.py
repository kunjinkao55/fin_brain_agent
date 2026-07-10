"""
FinBrain 数据工具层 — 纯函数，不依赖任何 Agent 框架。
可直接被 MCP Server、LangGraph Agent、或命令行脚本调用。
"""

import urllib.request
import urllib.parse
import ssl
import re
import json
import pandas as pd
import akshare as ak


# ============================================================
#  工具1：实时行情
# ============================================================

def fetch_stock_price(symbol: str) -> dict:
    """通过新浪财经 API 查询A股实时价格"""
    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        return {"error": f"无法识别的股票代码: {symbol}"}

    url = f"https://hq.sinajs.cn/list={full_code}"

    try:
        req = urllib.request.Request(url)
        req.add_header("Referer", "https://finance.sina.com.cn")
        req.add_header("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            text = resp.read().decode("gbk")

        match = re.search(r'"([^"]+)"', text)
        if not match:
            return {"error": f"未找到股票数据: {text[:100]}"}

        fields = match.group(1).split(",")
        if len(fields) < 32:
            return {"error": f"返回数据字段不足: {len(fields)}"}

        return {
            "name": fields[0],
            "open": fields[1],
            "yesterday_close": fields[2],
            "price": fields[3],
            "high": fields[4],
            "low": fields[5],
            "time": f"{fields[30]} {fields[31]}",
        }

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}


# ============================================================
#  工具2：历史K线
# ============================================================

def fetch_stock_history(symbol: str, scale: int = 240, datalen: int = 30) -> dict:
    """通过新浪API查询A股历史K线数据"""
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
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        req.add_header("Referer", "https://finance.sina.com.cn")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
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
    """获取近三年三大报表（东方财富 datacenter API）"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent":
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"

        def fetch_report(report_name):
            params = {
                "reportName": report_name,
                "columns": "ALL",
                "filter": f'(SECURITY_CODE="{symbol}")(DATE_TYPE_CODE="001")',
                "pageNumber": "1",
                "pageSize": "3",
                "sortColumns": "REPORT_DATE",
                "sortTypes": "-1",
                "source": "SECURITIES",
                "client": "PC",
            }
            url = base + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
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
            "PARENT_NETPROFIT": "净利润", "DEDUCT_PARENT_NETPROFIT": "扣非净利润",
            "SALE_EXPENSE": "销售费用", "MANAGE_EXPENSE": "管理费用",
            "FINANCE_EXPENSE": "财务费用", "OPERATE_PROFIT": "营业利润",
            "INCOME_TAX": "所得税",
        }
        CASHFLOW_COLS = {
            "NETCASH_OPERATE": "经营现金流净额", "NETCASH_INVEST": "投资现金流净额",
            "NETCASH_FINANCE": "筹资现金流净额", "CONSTRUCT_LONG_ASSET": "购建固定资产支付现金",
        }

        def pick(data, col_map):
            results = []
            for row in data:
                item = {"date": row["REPORT_DATE"][:10]}
                for eng_key, cn_name in col_map.items():
                    item[cn_name] = row.get(eng_key, None)
                results.append(item)
            return results

        return {
            "symbol": symbol,
            "balance":  pick(balance_data,  BALANCE_COLS),
            "profit":   pick(profit_data,   INCOME_COLS),
            "cashflow": pick(cashflow_data, CASHFLOW_COLS),
            "name": balance_data[0].get("SECURITY_NAME_ABBR", ""),
        }

    except Exception as e:
        return {"error": f"获取财报失败: {str(e)}"}


# ============================================================
#  工具4：估值指标
# ============================================================

def get_valuation(symbol: str) -> dict:
    """获取估值数据（ROE/毛利率/净利率/EPS等）"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent":
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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

        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
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

        return {
            "symbol": symbol,
            "data": results,
            "name": data["result"]["data"][0].get("SECURITY_NAME_ABBR", ""),
        }

    except Exception as e:
        return {"error": f"获取估值失败: {str(e)}"}


# ============================================================
#  工具5：行业信息
# ============================================================

def get_industry_info(symbol: str) -> dict:
    """获取个股所属行业 + 行业指数近期表现"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent":
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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

        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
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

        return {
            "symbol": symbol,
            "name": row.get("SECURITY_NAME_ABBR", ""),
            "industry_name": industry_name,
            "industry_code": row.get("INDUSTRY_CODE", ""),
            "index_trend": index_data,
        }

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
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent":
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        # 新浪全市场行情接口（分4页覆盖沪深A股）
        all_stocks = []
        seen = set()

        for page in range(1, 5):
            url = (f"http://vip.stock.finance.sina.com.cn/quotes_service/api/"
                   f"json_v2.php/Market_Center.getHQNodeData?"
                   f"page={page}&num=2000&sort=code&asc=1&node=hs_a")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
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
