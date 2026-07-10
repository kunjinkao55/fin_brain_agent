from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import mcp.types as types
import urllib.request
import ssl
import re
import json
import pandas as pd
import akshare as ak
import urllib.parse

app = Server("my-mcp-server")


def fetch_stock_price(symbol: str) -> dict:
    """通过新浪财经 API 查询A股实时价格（免费，无需注册）"""
    # 根据股票代码判断交易所前缀：60x/68x → sh, 00x/30x → sz
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
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        # 创建兼容性更好的 SSL 上下文，解决 Windows 11 上 SSL 握手失败问题
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            text = resp.read().decode("gbk")

        # 新浪返回格式: var hq_str_sh601991="名字,今开,昨收,现价,最高,最低,..."
        match = re.search(r'"([^"]+)"', text)
        if not match:
            return {"error": f"未找到股票数据: {text[:100]}"}

        fields = match.group(1).split(",")
        if len(fields) < 32:
            return {"error": f"返回数据字段不足: {len(fields)}"}

        return {
            "name": fields[0],           # 股票名称
            "open": fields[1],           # 今日开盘价
            "yesterday_close": fields[2], # 昨日收盘价
            "price": fields[3],          # 当前价格
            "high": fields[4],           # 今日最高
            "low": fields[5],            # 今日最低
            "time": f"{fields[30]} {fields[31]}",  # 日期 时间
        }

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}

def fetch_stock_history(symbol: str, scale: int = 240, datalen: int = 30) -> dict:
    """通过新浪API查询A股历史K线数据"""
    # 1. 判断交易所前缀（复制 fetch_stock_price 的逻辑）
    if symbol.startswith(("60", "68")):
        full_code = f"sh{symbol}"
    elif symbol.startswith(("00", "30")):
        full_code = f"sz{symbol}"
    else:
        return {"error": f"无法识别的股票代码: {symbol}"}

    # 2. 拼接URL（注意是 http，不是 https）
    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={full_code}&scale={scale}&ma=no&datalen={datalen}"
    )

    try:
        # 3. 发送请求（和查现价一样的 headers + SSL 配置）
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        req.add_header("Referer", "https://finance.sina.com.cn")

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            # 4. 这个接口返回UTF-8编码的JSON
            text = resp.read().decode("utf-8")
            data = json.loads(text)        # ← 记得在文件顶部 import json

        # 5. 整理返回数据，只保留画K线需要的字段
        klines = []
        for item in data:
            klines.append({
                "day": item["day"],         # 日期/时间
                "open": item["open"],       # 开盘价
                "high": item["high"],       # 最高价
                "low": item["low"],         # 最低价
                "close": item["close"],     # 收盘价
                "volume": item["volume"],   # 成交量
            })

        return {"data": klines}

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}

def get_financial_statements(symbol: str) -> dict:
    """获取近三年三大报表关键指标 + 主营构成"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        
        def fetch_report(report_name):
            """请求一个报表，返回解析后的JSON数据list"""
            # 拼URL参数
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
            query = urllib.parse.urlencode(params)       # 把字典转成 ?key=value&key=value...
            url = base + "?" + query

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["result"]["data"]            # 返回 [dict, dict, ...]

        # 分别请求三张表
        balance_data  = fetch_report("RPT_DMSK_FN_BALANCE")
        profit_data   = fetch_report("RPT_DMSK_FN_INCOME")
        cashflow_data = fetch_report("RPT_DMSK_FN_CASHFLOW")
            
        # ===== 列名映射 =====
        BALANCE_COLS = {
            "TOTAL_ASSETS":       "资产总计",
            "TOTAL_LIABILITIES":  "负债合计",
            "TOTAL_EQUITY":       "股东权益",
            "FIXED_ASSET":        "固定资产",
            "MONETARYFUNDS":      "货币资金",
            "ACCOUNTS_RECE":      "应收账款",
            "INVENTORY":          "存货",
            "ACCOUNTS_PAYABLE":   "应付账款",
            "DEBT_ASSET_RATIO":   "资产负债率",
        }

        INCOME_COLS = {
            "TOTAL_OPERATE_INCOME":   "营业总收入",
            "OPERATE_COST":           "营业成本",
            "PARENT_NETPROFIT":       "净利润",
            "DEDUCT_PARENT_NETPROFIT":"扣非净利润",
            "SALE_EXPENSE":           "销售费用",
            "MANAGE_EXPENSE":         "管理费用",
            "FINANCE_EXPENSE":        "财务费用",
            "OPERATE_PROFIT":         "营业利润",
            "INCOME_TAX":             "所得税",
        }

        CASHFLOW_COLS = {
            "NETCASH_OPERATE":       "经营现金流净额",
            "NETCASH_INVEST":        "投资现金流净额",
            "NETCASH_FINANCE":       "筹资现金流净额",
            "CONSTRUCT_LONG_ASSET":  "购建固定资产支付现金",
        }

        # ===== 提取字段（数据已经是 list[dict]，直接用） =====
        def pick(data, col_map):
            """从JSON数据中提取指定列，转成 [{date, 中文名, ...}]"""
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
            "name": balance_data[0].get("SECURITY_NAME_ABBR", ""),  # 从第一行取股票名称
        }

    except Exception as e:
        return {"error": f"获取财报失败: {str(e)}"}

def get_industry_info(symbol: str) -> dict:
    """获取行业信息"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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
        query = urllib.parse.urlencode(params)       # 把字典转成 ?key=value&key=value...
        url = base + "?" + query

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        row = data["result"]["data"][0]

        industry_name = row.get("INDUSTRY_NAME", "")

        # 获取行业指数近期行情（THS数据源，截至2024年初）
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
                        chg = (close - prev_close) / prev_close * 100
                        item["涨跌幅"] = f"{chg:+.2f}%"
                    else:
                        item["涨跌幅"] = "-"
                    prev_close = close
                    index_data.append(item)
                if index_data:
                    index_data = index_data[-5:]  # 去掉第一天（用于计算首日涨跌）
            except Exception:
                pass  # 行业指数获取失败不阻塞主流程

        return {
            "symbol": symbol,
            "name": row.get("SECURITY_NAME_ABBR", ""),
            "industry_name": industry_name,
            "industry_code": row.get("INDUSTRY_CODE", ""),
            "index_trend": index_data,
        }

    except Exception as e:
        return {"error": f"获取行业信息失败: {str(e)}"}

def get_valuation(symbol: str) -> dict:
    """获取估值数据"""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        base = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
        
        def fetch_report(report_name):
            """请求一个报表，返回解析后的JSON数据list"""
            # 拼URL参数
            params = {
                "reportName": report_name,
                "columns": "ALL",
                "filter": f'(SECURITY_CODE="{symbol}")',
                "pageNumber": "1",
                "pageSize": "3",
                "sortColumns": "REPORT_DATE",
                "sortTypes": "-1",
                "source": "SECURITIES",
                "client": "PC",
            }
            query = urllib.parse.urlencode(params)       # 把字典转成 ?key=value&key=value...
            url = base + "?" + query

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["result"]["data"]            # 返回 [dict, dict, ...]

        value_data  = fetch_report("RPT_F10_FINANCE_MAINFINADATA")
            
        # ===== 列名映射 =====
        VALUATION_COLS = {
        "REPORT_DATE_NAME": "报告期",
        "ROEJQ":            "ROE(%)",
        "XSMLL":            "毛利率(%)",
        "XSJLL":            "净利率(%)",
        "EPSJB":            "每股收益",
        "BPS":              "每股净资产",
        "TOTAL_SHARE":      "总股本",
        "PARENTNETPROFIT":  "归母净利润",
        "TOTALOPERATEREVE": "营业总收入",
        "ZCFZL":            "资产负债率(%)",
        }


        # ===== 提取字段（数据已经是 list[dict]，直接用） =====
        def pick(data, col_map):
            """从JSON数据中提取指定列，转成 [{date, 中文名, ...}]"""
            results = []
            for row in data:
                item = {"日期": row["REPORT_DATE"][:10]}
                for eng_key, cn_name in col_map.items():
                    item[cn_name] = row.get(eng_key, None)
                results.append(item)
            return results

        return {
            "symbol": symbol,
            "valuation": pick(value_data, VALUATION_COLS),
            "name": value_data[0].get("SECURITY_NAME_ABBR", ""),  # 从第一行取股票名称
        }

    except Exception as e:
        return {"error": f"获取估值失败: {str(e)}"}

def get_fund_flow(symbol: str, days: int = 30) -> dict:
    """获取个股资金流向，还没做完接口有反爬"""
    try:
        # 拼 secid
        if symbol.startswith(("60", "68")):
            secid = f"1.{symbol}"
        elif symbol.startswith(("00", "30")):
            secid = f"0.{symbol}"
        else:
            return {"error": f"无法识别: {symbol}"}

        url = (
            f"https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
            f"?secid={secid}&fields1=f1,f2,f3,f7"
            f"&fields2=f51,f52,f53,f54,f55,f56,f57&lmt={days}"
        )

        # 请求
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        req.add_header("Referer", "https://data.eastmoney.com/")   # ← 加这行
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # 解析：每条是 "日期,主力,超大单,大单,中单,小单,占比"
        klines = data["data"]["klines"]
        results = []
        for line in klines:
            parts = line.split(",")
            results.append({
                "日期":     parts[0],
                "主力净流入":  float(parts[1]),
                "超大单净流入": float(parts[2]),
                "大单净流入":  float(parts[3]),
                "中单净流入":  float(parts[4]),
                "小单净流入":  float(parts[5]),
                "主力净流入占比": parts[6] + "%",
            })

        return {"data": results, "symbol": symbol}

    except urllib.error.URLError as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except Exception as e:
        return {"error": f"查询异常: {str(e)}"}

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_stock_price",
            description="获取股票实时价格",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_stock_history",
            description="获取股票历史k线数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码"},
                    "scale":{
                        "type":"integer",
                        "description": "K线周期: 5/15/30/60(分钟), 240(日线), 默认240",
                        "default": 240
                    },
                    "datalen": {
                        "type": "integer",
                        "description": "返回数据条数，默认30",
                        "default": 30
                    }   
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_financial_statements",
            description="获取股票近三年资产负债表、利润表、现金流量表的关键科目",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 601991"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_fund_flow",
            description="获取个股资金流向",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 601991"},
                    "days": {
                        "type": "integer",
                        "description": "查询资金流向的天数，默认30",
                        "default": 30
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_valuation",
            description="获取股票估值数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 601991"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_industry_info",
            description="获取股票所属行业信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如 601991"}
                },
                "required": ["symbol"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_stock_price":
        symbol = arguments["symbol"]
        result = fetch_stock_price(symbol)

        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        text = (
            f"【{result['name']}】{symbol}\n"
            f"💰 最新价: {result['price']} 元\n"
            f"📈 今开: {result['open']} | 昨收: {result['yesterday_close']}\n"
            f"📊 最高: {result['high']} | 最低: {result['low']}\n"
            f"🕐 更新时间: {result['time']}"
        )
        return [TextContent(type="text", text=text)]
    elif name == "get_stock_history":
        symbol = arguments["symbol"]
        scale = arguments.get("scale", 240)
        datalen = arguments.get("datalen", 30)
        result = fetch_stock_history(symbol, scale, datalen)

        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        # 格式化成易读文本
        lines = [f"📈 K线数据 ({symbol}) - 共{len(result['data'])}条"]
        for k in result["data"]:
            lines.append(
                f"  {k['day']} | O:{k['open']} H:{k['high']} L:{k['low']} C:{k['close']} V:{k['volume']}"
            )
        return [TextContent(type="text", text="\n".join(lines))]
    elif name == "get_financial_statements":
        symbol = arguments["symbol"]
        result = get_financial_statements(symbol)

        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        name = result.get("name", symbol)
        text = f"【{name}】{symbol} 近三年财报关键数据\n\n"

        text += "📊 利润表:\n"
        for item in result["profit"]:
            text += f"  {item['date']} | "
            text += f"营收:{item.get('营业总收入','-')} | "       # ← 营业总收入
            text += f"成本:{item.get('营业成本','-')} | "
            text += f"净利润:{item.get('净利润','-')} | "
            text += f"扣非:{item.get('扣非净利润','-')}\n"

        text += "\n📋 资产负债表:\n"
        for item in result["balance"]:
            text += f"  {item['date']} | "
            text += f"总资产:{item.get('资产总计','-')} | "
            text += f"负债:{item.get('负债合计','-')} | "
            text += f"权益:{item.get('股东权益','-')} | "         # ← 股东权益
            text += f"负债率:{item.get('资产负债率','-')}\n"

        text += "\n💰 现金流:\n"
        for item in result["cashflow"]:
            text += f"  {item['date']} | "
            text += f"经营:{item.get('经营现金流净额','-')} | "
            text += f"投资:{item.get('投资现金流净额','-')} | "
            text += f"筹资:{item.get('筹资现金流净额','-')} | "
            text += f"购建固定资产:{item.get('购建固定资产支付现金','-')}\n"

        return [TextContent(type="text", text=text)]
    elif name == "get_fund_flow":
        symbol = arguments["symbol"]
        days = arguments.get("days", 30)
        result = get_fund_flow(symbol, days)
        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        text = f"【{result['symbol']}】资金流向\n"
        for line in result["data"]:
            text += f"  {line['日期']} | 主力净流入: {line['主力净流入']} | "
            text += f"超大单净流入: {line['超大单净流入']} | "
            text += f"大单净流入: {line['大单净流入']} | "
            text += f"中单净流入: {line['中单净流入']} | "
            text += f"小单净流入: {line['小单净流入']} | "
            text += f"主力净流入占比: {line['主力净流入占比']}\n"

        return [TextContent(type="text", text=text)]
    elif name == "get_valuation":
        symbol = arguments["symbol"]
        result = get_valuation(symbol)
        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        text = f"【{result['symbol']}】估值数据\n"
        for line in result["valuation"]:
            text += f"  {line['日期']} | "
            text += f"ROE:{line.get('ROE(%)','-')}% | "
            text += f"毛利率:{line.get('毛利率(%)','-')}% | "
            text += f"净利率:{line.get('净利率(%)','-')}% | "
            text += f"每股收益:{line.get('每股收益','-')} | "
            text += f"资产负债率:{line.get('资产负债率(%)','-')}%\n"

        return [TextContent(type="text", text=text)]
    elif name == "get_industry_info":
        symbol = arguments["symbol"]
        result = get_industry_info(symbol)
        if "error" in result:
            return [TextContent(type="text", text=f"❌ {result['error']}")]

        text = f"【{result['name']}】{result['symbol']} 行业信息\n"
        text += f"  行业: {result['industry_name']} | 代码: {result['industry_code']}\n"

        if result.get("index_trend"):
            text += f"\n📊 {result['industry_name']}行业指数 近5日:\n"
            for item in result["index_trend"]:
                change = item.get("涨跌幅", "-")
                text += f"  {item['日期']} | 收:{item['收盘']} | 涨跌:{change}\n"

        return [TextContent(type="text", text=text)]
    else:
        return [TextContent(type="text", text=f"❌ 未知工具: {name}")]
    
async def main():
    async with stdio_server() as streams:
        await app.run(*streams, initialization_options=app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    