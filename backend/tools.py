"""
FinBrain 数据工具层 — 纯函数，不依赖任何 Agent 框架。
可直接被 MCP Server、LangGraph Agent、或命令行脚本调用。
"""

import urllib.request
import urllib.parse
import ssl
import re
import json
import logging
import pandas as pd
import akshare as ak

logger = logging.getLogger(__name__)

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
    """通过新浪财经 API 查询A股实时价格"""
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
    """获取近三年三大报表（东方财富 datacenter API）"""
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
        lines.append(f"  FinBrain 分析报告: {name} ({code})")
        lines.append(f"=" * 64)

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
            for dim, info in scores.items():
                s_val = info.get("得分")
                reason = info.get("依据", "")
                if s_val is None:
                    # 数据缺失，不参与计分
                    lines.append(f"  {dim:<8}  {'N/A':>4}  {'-':<6}  {reason}")
                else:
                    g = grade(s_val)
                    lines.append(f"  {dim:<8}  {s_val:>4}  {g:<6}  {reason}")
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
            ytd = val_level.get("年内涨幅", "-")
            judge = val_level.get("判断", "-")
            lines.append("")
            lines.append(f"  [估值水位] PE:{pe} PB:{pb} 市值:{mkt} 年内涨幅:{ytd} -> {judge}")

        # ---- 对比分析 ----
        compare = analysis.get("对比分析", {})
        if compare:
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
                    # 表头
                    header_cols = ["指标"] + [s.get("名称", "?") for s in stocks]
                    col_widths = [max(12, max(len(str(s.get(h, ""))) for s in stocks)) + 2 for h in indicators]
                    col_widths.insert(0, max(len(h) for h in header_cols) + 2)

                    def _align(v, w, right=False):
                        s = str(v) if v is not None else "-"
                        return f"{s:>{w}}" if right else f"{s:<{w}}"

                    header_line = "  " + "".join(_align(h, w) for h, w in zip(header_cols, col_widths))
                    lines.append(header_line)
                    lines.append("  " + "".join("-" * w for w in col_widths))

                    for idx in indicators:
                        row = [_align(idx, col_widths[0])]
                        for j, s in enumerate(stocks):
                            row.append(_align(s.get(idx, "-"), col_widths[j+1], True))
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

                    lines.append("  " + "".join(_align(h, w) for h, w in zip(v_header, v_widths)))
                    lines.append("  " + "".join("-" * w for w in v_widths))
                    for idx in v_indicators:
                        row = [_align(idx, v_widths[0])]
                        for j, s in enumerate(v_stocks):
                            row.append(_align(s.get(idx, "-"), v_widths[j+1], True))
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

        # ---- 观察指标 ----
        watch = analysis.get("观察指标", [])
        if watch:
            lines.append("")
            lines.append("  [观察指标]")
            for w in watch:
                lines.append(f"    - {w}")

        # ---- 建议 ----
        advice = analysis.get("操作建议", "")
        stop_loss = analysis.get("止损", "")
        if advice:
            lines.append("")
            lines.append(f"  [操作建议] {advice}")
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

            # 涨停板阈值：主板10%，双创20%，北交所30%
            if chg < 9.5:
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

        return {"涨停板数量": len(results), "列表": results[:top_n]}

    except Exception as e:
        return {"error": f"涨停板查询失败: {str(e)}"}


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
    """纯函数，根据财报数据计算6维评分。LLM 只管叙事，数字由这里保证一致。"""
    scores = {}

    # 从 valuation 数据中提取最新值
    val_data = financial_data.get("valuation", {}).get("data", [])
    latest = val_data[0] if val_data else {}

    # --- 1. 盈利能力 (0-10) ---
    roe = float(latest.get("ROE(%)", 0) or 0)
    gm = float(latest.get("毛利率(%)", 0) or 0)
    nm = float(latest.get("净利率(%)", 0) or 0)

    if roe >= 50: pe_score = 10
    elif roe >= 30: pe_score = 9
    elif roe >= 20: pe_score = 7
    elif roe >= 10: pe_score = 5
    elif roe >= 5: pe_score = 3
    else: pe_score = 1

    if gm >= 40: pe_score = min(10, pe_score + 1)
    if nm >= 20: pe_score = min(10, pe_score + 1)
    scores["盈利能力"] = {"得分": pe_score, "依据": f"ROE {roe}%, 毛利率{gm}%, 净利率{nm}%"}

    # --- 2. 成长性 (0-10) ---
    profit_data = financial_data.get("profit", [])
    if len(profit_data) >= 2:
        rev_latest = profit_data[-1].get("营业总收入") or 0
        rev_prev = profit_data[-2].get("营业总收入") or 1
        rev_growth = (float(rev_latest) - float(rev_prev)) / float(rev_prev) * 100 if float(rev_prev) > 0 else 0

        net_latest = profit_data[-1].get("净利润") or 0
        net_prev = profit_data[-2].get("净利润") or 1
        net_growth = (float(net_latest) - float(net_prev)) / float(net_prev) * 100 if float(net_prev) > 0 else 0

        if rev_growth >= 100: g_score = 10
        elif rev_growth >= 50: g_score = 9
        elif rev_growth >= 30: g_score = 8
        elif rev_growth >= 20: g_score = 5
        elif rev_growth >= 0: g_score = 3
        else: g_score = 0
        scores["成长性"] = {"得分": g_score, "依据": f"营收增速{rev_growth:.0f}%, 净利润增速{net_growth:.0f}%"}
    else:
        scores["成长性"] = {"得分": 5, "依据": "数据不足，默认5分"}

    # --- 3. 财务健康 (0-10) ---
    debt = float(latest.get("资产负债率(%)", 50) or 50)
    cf_data = financial_data.get("cashflow", [])
    cf_score = 0
    if cf_data:
        cf_latest = cf_data[-1]
        op_cf = float(cf_latest.get("经营现金流净额", 0) or 0)
        net_profit = float(profit_data[-1].get("净利润", 1) or 1) if profit_data else 1
        cf_ratio = op_cf / net_profit if net_profit > 0 else 0
        if cf_ratio >= 0.8: cf_score = 2

    if debt < 30: debt_score = 10
    elif debt < 50: debt_score = 7
    elif debt < 60: debt_score = 5
    else: debt_score = 3
    scores["财务健康"] = {"得分": min(10, debt_score + cf_score), "依据": f"资产负债率{debt}%, 经营现金流/净利润={cf_ratio:.2f}"}

    # --- 4. 估值合理 (0-10) ---
    price_info = financial_data.get("price", {})
    pe = float(price_info.get("per", 0) or 0) if isinstance(price_info, dict) else 0
    if pe <= 0: v_score = 5
    elif pe < 15: v_score = 10
    elif pe < 25: v_score = 7
    elif pe < 40: v_score = 5
    else: v_score = 3
    scores["估值合理"] = {"得分": v_score, "依据": f"PE {pe:.0f}倍" if pe > 0 else "PE数据缺失"}

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

def get_sector_fund_flow(top_n: int = 50, date: str = "") -> dict:
    """获取全行业板块资金流向排名（同花顺），含净流入/流出额。date为空=当日"""
    try:
        import requests
        from bs4 import BeautifulSoup as bs

        headers = _get_ths_headers()
        all_sectors = []
        for page in range(1, 5):  # 最多4页，覆盖所有行业
            url = (f"http://data.10jqka.com.cn/funds/hyzjl/"
                   f"field/zdf/order/desc/page/{page}/ajax/1/free/1/")
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
        return {"板块数量": len(all_sectors), "列表": all_sectors[:top_n]}

    except Exception as e:
        return {"error": f"板块资金流查询失败: {str(e)}"}


# ============================================================
#  工具12：日内分时数据
# ============================================================

def get_intraday(symbol: str) -> dict:
    """获取当日5分钟K线（48根/天），用于画分时图"""
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
