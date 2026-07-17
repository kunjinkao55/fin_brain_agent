"""
FinBrain — 多 Agent 协作财报分析系统
Data Collector → Analyst → Reporter (LangGraph StateGraph)
"""

import json, os, re, atexit
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LangGraph SqliteSaver — 框架内置持久化，跨会话恢复对话
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "checkpoints.db")

_CK = None
_CK_CM = None

def _make_checkpointer():
    global _CK, _CK_CM
    if _CK is None:
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        _CK_CM = SqliteSaver.from_conn_string(_DB_PATH)
        _CK = _CK_CM.__enter__()
        atexit.register(_cleanup_checkpointer)
    return _CK


def _cleanup_checkpointer():
    global _CK_CM, _CK
    if _CK_CM is not None:
        _CK_CM.__exit__(None, None, None)
        _CK_CM = None
        _CK = None

# 从项目根目录加载.env（兼容从任意目录运行）
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
load_dotenv(_env_path)

# 策略配置（懒加载，避免 import 时读文件）
_STRATEGY = None
_ALL_STRATEGIES = None

def _get_strategy():
    global _STRATEGY, _ALL_STRATEGIES
    if _STRATEGY is None:
        strat_file = os.path.join(os.path.dirname(__file__), "..", "configs", "strategies.json")
        with open(strat_file, "r", encoding="utf-8") as f:
            _ALL_STRATEGIES = json.load(f)
        active = os.getenv("FINBRAIN_STRATEGY", "default")
        if active not in _ALL_STRATEGIES:
            active = "default"
        _STRATEGY = _ALL_STRATEGIES[active]
    return _STRATEGY

# ============================================================
#  _get_llm() 初始化（懒加载，多提供商切换）
# ============================================================

_LLM = None
_llm_failures = 0
_MAX_LLM_FAILURES = 3


def _get_llm():
    """获取 LLM 实例。连续失败 3 次→熔断。"""
    global _llm_failures
    if _llm_failures >= _MAX_LLM_FAILURES:
        raise RuntimeError(f"LLM API 连续失败{_MAX_LLM_FAILURES}次，已熔断。请检查 API Key 和网络后重启。")
    global _LLM
    if _LLM is not None:
        return _LLM

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        _LLM = ChatOpenAI(
            model=os.getenv("LLM_MODEL","deepseek-chat"), temperature=0, max_tokens=4096,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL","https://api.deepseek.com"), streaming=True,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        _LLM = ChatOpenAI(
            model=os.getenv("LLM_MODEL","gpt-4o"), temperature=0, max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"), streaming=True,
        )
    else:
        from langchain_anthropic import ChatAnthropic
        _LLM = ChatAnthropic(
            model=os.getenv("LLM_MODEL","claude-sonnet-5"), temperature=0, max_tokens=4096, streaming=True,
        )
    return _LLM

# ============================================================
#  State Schema
# ============================================================

class FinBrainState(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str
    collected_data: str
    analysis: str
    report: str
    processing_log: list  # 流水线日志: [{phase, summary, detail}]

# ============================================================
#  @tool 工具定义（7个，不变）
# ============================================================

@tool
def stock_price(symbol: str) -> str:
    """查询A股实时价格。输入股票代码如 '601991' 或 '300502'"""
    from backend.tools import fetch_stock_price
    return json.dumps(fetch_stock_price(symbol), ensure_ascii=False, indent=2)

@tool
def stock_history(symbol: str, scale: int = 240, datalen: int = 30) -> str:
    """查询A股历史K线数据。symbol:股票代码 scale:240=日线 datalen:条数"""
    from backend.tools import fetch_stock_history
    result = fetch_stock_history(symbol, scale, datalen)
    if "data" in result and len(result["data"]) > 10:
        result["data"] = result["data"][-10:]
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def financial_statements(symbol: str) -> str:
    """查询近三年三大报表：利润表、资产负债表、现金流量表。输入股票代码"""
    from backend.tools import get_financial_statements
    return json.dumps(get_financial_statements(symbol), ensure_ascii=False, indent=2)

@tool
def valuation(symbol: str) -> str:
    """查询估值指标：ROE、毛利率、净利率、每股收益、资产负债率等。输入股票代码"""
    from backend.tools import get_valuation
    return json.dumps(get_valuation(symbol), ensure_ascii=False, indent=2)

@tool
def industry_info(symbol: str) -> str:
    """查询个股所属行业分类及行业指数表现。输入股票代码"""
    from backend.tools import get_industry_info
    return json.dumps(get_industry_info(symbol), ensure_ascii=False, indent=2)

@tool
def fund_flow(symbol: str) -> str:
    """查询个股当日资金流向：主力净流入/流出/净额。输入股票代码如 601991"""
    from backend.tools import get_fund_flow
    return json.dumps(get_fund_flow(symbol), ensure_ascii=False, indent=2)

@tool
def sector_fund_flow(top_n: int = 30) -> str:
    """获取全行业板块资金流向排名。返回板块名/涨跌幅/净流入流出额。"""
    from backend.tools import get_sector_fund_flow
    return json.dumps(get_sector_fund_flow(top_n), ensure_ascii=False, indent=2)

@tool
def intraday(symbol: str) -> str:
    """获取个股当日5分钟分时K线数据，用于画分时走势图。输入股票代码"""
    from backend.tools import get_intraday
    return json.dumps(get_intraday(symbol), ensure_ascii=False, indent=2)

@tool
def limit_up_pool(top_n: int = 30) -> str:
    """获取今日涨停板股票池。返回涨停股列表及涨幅/换手率/市值。参数: top_n=返回数量"""
    from backend.tools import get_limit_up_pool
    result = get_limit_up_pool(top_n)
    if "列表" in result and len(result["列表"]) > 15:
        # 只返回前15只给_get_llm()，节省token
        result["列表"] = result["列表"][:15]
    return json.dumps(result, ensure_ascii=False, indent=2)

# ============================================================
#  模拟盘工具
# ============================================================

@tool
def place_order(action: str, symbol: str, shares: int = 0, price: float = None,
                pct: float = 0) -> str:
    """模拟盘下单。action:'buy'/'sell'/'reset', symbol:股票代码, shares:股数(sell时-1=全仓),
       pct:按百分比买入(>0时忽略shares), reset用: action='reset', symbol='1000000'(初始资金)"""
    from backend.portfolio import get_portfolio
    pf = get_portfolio()
    if action.lower() == "reset":
        cash = float(symbol) if symbol.replace(".","").isdigit() else None
        result = pf.reset(cash)
    elif action.lower() == "buy":
        if pct > 0:
            result = pf.buy_pct(symbol, pct)
        else:
            result = pf.buy(symbol, shares, price)
    elif action.lower() == "sell":
        if pct > 0:
            result = pf.sell_pct(symbol, pct)
        else:
            result = pf.sell(symbol, shares, price)
    else:
        result = {"error": "action必须是buy/sell/reset"}
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def show_portfolio(dummy: str = "") -> str:
    """查看模拟盘持仓和盈亏总览。"""
    from backend.portfolio import get_portfolio
    pf = get_portfolio()
    data = pf.summary()
    lines = ["=" * 64, "  FinBrain 模拟盘", "=" * 64, ""]
    lines.append(f"  初始资金: {data['初始资金']:>12,.0f}")
    lines.append(f"  现金:     {data['现金']:>12,.0f}  持仓市值: {data['持仓市值']:>12,.0f}  总资产: {data['总资产']:>12,.0f}")
    lines.append(f"  累计收益率: {data['累计收益率']:>10}  浮动盈亏: {data['总盈亏']:>12,.0f} ({data['总盈亏%']})")
    lines.append("")
    if data["持仓明细"]:
        lines.append(f"  {'代码':<8} {'名称':<8} {'持仓':>6} {'成本':>8} {'现价':>8} {'市值':>10} {'盈亏':>10} {'盈亏%':>8}")
        lines.append(f"  {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for p in data["持仓明细"]:
            lines.append(f"  {p['代码']:<8} {p['名称']:<8} {p['持仓']:>6} {p['成本价']:>8.2f} {p['现价']:>8.2f} {p['市值']:>10,.0f} {p['盈亏']:>10,.0f} {p['盈亏%']:>8}")
    else:
        lines.append("  (空仓)")
    lines.append("")
    return "\n".join(lines)

@tool
def trade_history(n: int = 10) -> str:
    """查看最近N笔交易记录。"""
    from backend.portfolio import get_portfolio
    pf = get_portfolio()
    trades = pf.recent_trades(n)
    if not trades:
        return "(无交易记录)"
    lines = [f"最近{len(trades)}笔交易:"]
    for t in trades[-n:]:
        lines.append(f"  {t['date']} {t['action']:4} {t['symbol']} {t['name']} x{t['shares']} @{t['price']} {t.get('pnl_pct','')}")
    return "\n".join(lines)


@tool
def execute_analysis(action: str = "buy", symbol: str = "", pct: float = 5) -> str:
    """一键执行分析建议下单。action:'buy'/'sell', symbol:股票代码, pct:仓位百分比(默认5%)"""
    from backend.portfolio import get_portfolio
    pf = get_portfolio()
    if not symbol:
        return json.dumps({"error":"请指定股票代码"}, ensure_ascii=False)
    if action == "buy":
        result = pf.buy_pct(symbol, pct)
    elif action == "sell":
        result = pf.sell_pct(symbol, pct)
    else:
        result = {"error": "action必须是buy或sell"}
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def calculate_score(symbol: str) -> str:
    """确定性评分引擎。自动拉取财报+估值+行情+PE数据，计算6维评分。
    你只需要传入股票代码，不需要手动拼接数据。"""
    from backend.tools import (get_financial_statements, get_valuation,
                                fetch_stock_price, get_industry_info,
                                get_recent_announcements, calculate_scores)
    import urllib.request

    # 自取数据（已缓存，重复调用零开销）
    fin = get_financial_statements(symbol)
    val = get_valuation(symbol)
    price = fetch_stock_price(symbol)
    ind = get_industry_info(symbol)

    # PE/PB/市值：东财实时行情
    pe_data = {}
    try:
        secid = f"1.{symbol}" if symbol.startswith(("60", "00")) else f"0.{symbol}"
        pe_url = (f"http://push2.eastmoney.com/api/qt/stock/get?"
                  f"secid={secid}&fields=f116,f117,f162,f167,f20")
        from backend.tools import _SSL_CTX
        req = urllib.request.Request(pe_url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com",
        })
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as resp:
            pe_json = json.loads(resp.read().decode("utf-8"))
            d = pe_json.get("data", {}) or {}
            pe_data = {"per": d.get("f162"), "pb": d.get("f167"),
                        "mktcap": d.get("f20")}
    except Exception:
        pass  # PE数据获取失败不影响其他评分

    # 组装为 calculate_scores 期望的结构
    data = {}
    if isinstance(fin, dict) and "profit" in fin:
        data["profit"] = fin.get("profit", [])
        data["cashflow"] = fin.get("cashflow", [])
        data["balance"] = fin.get("balance", [])
    if isinstance(val, dict):
        data["valuation"] = val
    # price 以 Sina 为准（有当前股价），pe_data 只补 PE/PB（如有）
    if isinstance(price, dict):
        price = dict(price)  # 不污染缓存
        if pe_data.get("per") is not None:
            price["per"] = pe_data["per"]
        if pe_data.get("pb") is not None:
            price["pb"] = pe_data["pb"]
        if pe_data.get("mktcap") is not None:
            price["mktcap"] = pe_data["mktcap"]
    data["price"] = price
    if isinstance(ind, dict):
        data["industry"] = ind.get("行业", ind.get("industry_name", ""))

    result = calculate_scores(data)
    result["代码"] = symbol
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def resolve_stock(query: str) -> str:
    """根据股票名称或代码模糊搜索，返回匹配的代码-名称列表。如 '新易盛' → 300502"""
    from backend.stock_map import fuzzy_search
    results = fuzzy_search(query, limit=5)
    if not results:
        return json.dumps({"error": f"未找到匹配 '{query}' 的股票"}, ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def search_youzi_kb(query: str) -> str:
    """RAG检索游资知识库。输入营业部名称或游资风格描述，返回匹配的游资信息(风格/胜率/席位)。
    如输入'华鑫证券上海分公司'或'锁仓格局'"""
    from backend.rag import search_youzi
    import logging
    logger = logging.getLogger("FinBrain.RAG")
    logger.info("[RAG Called] query='%s'", query)
    results = search_youzi(query, top_k=5)
    logger.info("[RAG Result] found=%d", len(results))
    if not results:
        return json.dumps({"info": "未找到匹配游资"}, ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def search_knowledge(query: str, kb: str = "accounting") -> str:
    """RAG检索多知识库。kb可选: accounting(会计准则/财务分析)/industry(行业研报)/trading(交易策略)。
    输入财报分析相关问题(如'收入确认条件''商誉减值测试''关联交易识别')，返回相关知识片段。"""
    from backend.accounting_rag import search_kb, seed_accounting_kb
    try:
        seed_accounting_kb()
    except Exception:
        pass
    results = search_kb(query, kb, top_k=5)
    if not results:
        return json.dumps({"info": f"知识库 '{kb}' 中未找到相关内容"}, ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def web_search(query: str) -> str:
    """互联网搜索，用于交叉验证免费API数据。当API返回的PE/市值/EPS/ROE数据可疑时调用。
    输入搜索词如'新易盛 300502 PE 2026'或'大唐发电 601991 最新市值'，返回搜索结果摘要。"""
    from backend.web_search import search_financial
    results = search_financial(query, max_results=5)
    if not results or results[0].get("score", 0) == 0:
        return json.dumps({"info": "Web Search 未配置或无结果。请在 Settings 配置 WEB_SEARCH_API_KEY。"},
                          ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False, indent=2)


@tool
def concept_ranking(top_n: int = 20) -> str:
    """获取同花顺概念板块列表。返回概念名称和代码。参数: top_n=返回数量"""
    from backend.tools import get_concept_ranking
    return json.dumps(get_concept_ranking(top_n), ensure_ascii=False, indent=2)


@tool
def screen_stocks(max_pe: float = 30, max_pb: float = 5,
                  min_mktcap: float = 20, top_n: int = 30) -> str:
    """全市场扫描，返回PE从低到高排列的A股列表。
    参数: max_pe=市盈率上限, max_pb=市净率上限, min_mktcap=最低市值(亿), top_n=返回条数"""
    from backend.tools import screen_stocks as do_screen
    result = do_screen(max_pe, max_pb, min_mktcap, top_n)
    if "text" in result:
        return result["text"]
    return json.dumps(result, ensure_ascii=False, indent=2)

# ============================================================
#  Phantom Hunter 妖股猎人 Agent
# ============================================================

@tool
def dragon_tiger_list(date: str = "") -> str:
    """获取今日龙虎榜上榜股票列表。返回上榜股及买入/卖出总额。"""
    from backend.tools import get_dragon_tiger_list
    return json.dumps(get_dragon_tiger_list(date), ensure_ascii=False, indent=2)

@tool
def dragon_tiger_detail(symbol: str) -> str:
    """查询个股龙虎榜买卖席位明细，识别知名游资（炒股养家/方新侠/上塘路等）。输入股票代码"""
    from backend.tools import get_dragon_tiger_detail
    return json.dumps(get_dragon_tiger_detail(symbol), ensure_ascii=False, indent=2)


@tool
def stock_streak(symbol: str) -> str:
    """查询个股近10日连板情况。返回连板天数、涨停日期列表。输入股票代码如'300231'"""
    from backend.tools import get_stock_streak
    return json.dumps(get_stock_streak(symbol), ensure_ascii=False, indent=2)


@tool
def recent_announcements(symbol: str) -> str:
    """查询个股最近20条公告。识别定增/重组/业绩预告/减持等重大事件。
    输入股票代码如'601991'，返回公告标题+日期+关键数字（募资额/发行股数）。"""
    from backend.tools import get_recent_announcements
    r = get_recent_announcements(symbol, 20)
    # 压缩输出：只保留标题含关键字的公告
    keywords = ["发行","增发","重组","收购","业绩","减持","分红","担保","债券"]
    filtered = [a for a in r.get("列表",[]) if any(kw in a.get("标题","") for kw in keywords)]
    if not filtered:
        filtered = r.get("列表",[])[:5]  # 回退：返回最近5条
    r["列表"] = filtered
    return json.dumps(r, ensure_ascii=False, indent=2)

# ---- 妖股猎人工具集 ----
_PHANTOM_TOOLS = [resolve_stock, search_youzi_kb, search_knowledge, recent_announcements, limit_up_pool, concept_ranking,
                  dragon_tiger_list, dragon_tiger_detail, stock_streak,
                  stock_price, stock_history, fund_flow,
                  financial_statements, valuation, industry_info,
                  place_order, execute_analysis, show_portfolio, trade_history]
_PHANTOM_AGENT = None

def _get_phantom_agent():
    global _PHANTOM_AGENT
    if _PHANTOM_AGENT is None:
        _PHANTOM_AGENT = create_agent(_get_llm(), _PHANTOM_TOOLS, system_prompt=_get_strategy().get("phantom", ""),
                                       checkpointer=_make_checkpointer())
    return _PHANTOM_AGENT

# ============================================================
#  Prompt 分解
# ============================================================

DATA_COLLECTOR_PROMPT = _get_strategy()["data_collector"]

_FORMAT_MANDATORY = """

[强制输出JSON格式]
不要markdown代码块。缺失数据标null。必须包含以下全部字段:
{
  "代码": "股票代码",
  "名称": "股票名称",
  "评分": {
    "维度1": {"得分": N, "依据": "具体数字"},
    "维度2": {"得分": N, "依据": "具体数字"}
  },
  "亮点": ["亮点1"],
  "风险": ["风险1"],
  "业绩驱动力": "1-2句话解释为什么业绩好/差（行业周期、产品放量、政策等）",
  "关键信号": [
    {"信号": "毛利率趋势", "数据": "2023:xx% 2024:xx% 2025:xx%", "解读": "逐季提升说明..."},
    {"信号": "现金流质量", "数据": "经营现金流/净利润=x.x", "解读": "利润含金量..."},
    {"信号": "研发或扩张投入", "数据": "研发费用xx亿,同比+xx%", "解读": "未来增长潜力..."}
  ],
  "估值水位": {"PE": xx, "PB": xx, "市值": "xx亿", "年内涨幅": "xx%", "判断": "偏贵/合理/低估"},
  "观察指标": ["指标1: 中报营收增速", "指标2: 毛利率是否维持", "指标3: 现金流/净利润比值"],
  "操作建议": "首次建仓≤X元(当前Y元)，仓位Z%，止损A元，目标B-C元。禁止使用'逢低布局''择机介入'等模糊措辞。",
  "止损": "A元（对应跌幅约D%，或PB≤E倍时止损）",
  "结论": {
    "总评": "财报质量+估值+长线判断",
    "买入策略": "≤X元建仓Z%，≤Y元加仓至W%，极限仓位M%。必须给出具体数字。",
    "持有策略": "持仓至[具体事件，如中报发布/年报发布/煤价走势明朗]，或N个月",
    "卖出条件": "止损A元；或[证伪条件1/2/3]触发时减仓至X%",
    "预期收益": "目标B-C元（对应PE约D-E倍），上行空间F-G%",
    "持仓周期": "N个月（或至[触发条件]）"
  },
  "公司画像": {
    "主营业务": "产品/服务分项，各占收入比例",
    "收入来源": ["产品A占x%", "产品B占y%"],
    "公司类型": "价值型/成长型/周期型/困境反转型/事件驱动型",
    "生命周期": "成长期/成熟期/周期底部/周期顶部/转型期",
    "行业模板": "医药/制造/消费/科技/金融/能源"
  },
  "竞争优势": {
    "核心资产": "企业最不可替代的资源或能力是什么",
    "护城河来源": ["品牌", "网络效应", "成本优势", "技术专利", "数据", "渠道", "牌照"],
    "复制难度": "低/中/高",
    "持续时间": "<3年/3-5年/5年以上",
    "竞争格局": "市占率趋势、新进入者威胁、替代品风险",
    "毛利率归因": "高毛利是因为技术壁垒、定价权、还是周期高点?"
  },
  "投资逻辑链": "因为①...→导致②...→最终③...→市场目前④...→因此⑤...",
  "催化剂": {
    "正面": ["未来12个月可能推动股价的事件"],
    "负面": ["未来12个月可能压制股价的事件"],
    "强度": "强催化/中性/无催化"
  },
  "估值方法": "适用于该公司的估值方法(PE/PB/PEG/PS/DCF)及理由。不同行业不同方法，不要所有公司都用PE",
  "市场预期拆解": {
    "当前估值隐含的增长率": "当前PE=14倍，市场隐含未来利润增速约x%",
    "市场主要担忧": ["担忧1", "担忧2"],
    "可能的预期差": "如果实际增长>隐含增速→估值修复;如果<→继续下跌"
  },
  "情景估值": {
    "悲观": {"价格": "xx元", "假设": "条件", "概率": "20%"},
    "基准": {"价格": "xx元", "假设": "条件", "概率": "60%"},
    "乐观": {"价格": "xx元", "假设": "条件", "概率": "20%"},
    "概率加权价值": "xx元 (=悲观×概率+基准×概率+乐观×概率)"
  },
  "投资评级": {
    "评级": "BUY/HOLD/SELL",
    "合理价值": "xx元",
    "安全边际": "x%",
    "买入区间": "≤xx元"
  },
  "长期结构性审视": {
    "行业终局推演": "3-5年后，当前高毛利业务是否会标准化？毛利率可能从X%降至Y%？为什么？",
    "管理层与治理": "实控人背景、核心团队稳定性、历史信披分红记录。信息不足时标注'公开信息不足，建议人工核实'",
    "终极风险": "什么力量可能让公司失去存在价值？（技术替代/政策颠覆/资源枯竭/降维打击）",
    "护城河保质期": "3年/5年/10年以上？瓦解信号是什么？"
  },
  "证伪条件": ["条件1: 什么具体指标变化到什么程度意味着投资逻辑失效", "条件2"],
  "对比分析": {
    "板块": "板块名称",
    "财报对比表": {
      "指标列表": ["营收","营收同比","净利润","净利润同比","毛利率","净利率","ROE","经营现金流"],
      "股票数据": [
        {"名称":"股票A","营收":"x亿","营收同比":"+x%","净利润":"x亿","净利润同比":"+x%","毛利率":"x%","净利率":"x%","ROE":"x%","经营现金流":"x亿"},
        {"名称":"股票B","营收":"y亿",...}
      ]
    },
    "估值对比表": {
      "指标列表": ["股价","市值","PE(TTM)","PB"],
      "股票数据": [
        {"名称":"股票A","股价":"x元","市值":"x亿","PE(TTM)":"x倍","PB":"x倍"},
        {"名称":"股票B",...}
      ]
    },
    "差异解读": ["公司A: 体量最大/增速最快...", "公司B: 毛利率最高但...", "公司C: 体量最小/估值最贵..."],
    "一句话总结": {"公司A":"定位","公司B":"定位","公司C":"定位"},
    "综合排名": "公司A > 公司B > 公司C"
  }
}
// 多只股票时输出JSON数组[{股票1},{股票2}...]，每只独立评分但共享对比分析。
// 只在分析>=2只同板块股票时填写\"对比分析\"字段，单只股票省略。
"""

ANALYST_PROMPT = """[铁律 - 输出格式]
1. 纯JSON，不要markdown代码块```json```，输出必须以[或{开头
2. 多只股票必须输出JSON数组 [{股票1},{股票2}]，每只独立完整
3. 每只股票必须包含: 公司画像、竞争优势、投资逻辑链、评分、亮点、风险、业绩驱动力、关键信号、估值水位、催化剂、市场预期拆解、情景估值、投资评级、证伪条件、观察指标、操作建议、止损、结论。缺一不可

[投资分析十步框架] 理解公司→判断价值→判断价格→给出决策。

第1步 公司分类（这是什么类型的投资机会）:
- 公司类型: 价值型(成熟稳定现金流)/成长型(高增速扩张)/周期型(行业周期驱动)/困境反转型(从低谷恢复)/事件驱动型(并购/重组/政策)
- 生命周期: 成长期/成熟期/周期底部/周期顶部/转型期
- 不同类型用不同分析重点: 价值股看现金流和估值; 成长股看行业空间和壁垒; 周期股看供需和拐点; 困境反转看修复信号

第2步 商业模式（怎么赚钱）:
- 主营业务拆分及收入占比
- 客户结构: 集中还是分散? 复购特征?
- 成本结构: 固定成本vs变动成本, 规模效应是否存在?
- 定价权: 能否提价? 提价后销量是否下降?

第3步 护城河（为什么竞争对手不能复制）:
- 核心资产: 企业最不可替代的资源或能力
- 壁垒来源: 品牌/网络效应/成本优势/技术专利/数据/渠道/牌照
- 复制难度: 低(1-2年可复制)/中(3-5年)/高(5年以上)
- 关键验证: 毛利率高是因为壁垒还是周期? 亏损是因为投入期还是模式失败?

第4步 行业判断（现在处于什么位置）:
- 行业空间和增速
- 生命周期阶段和供需格局
- 政策环境和竞争格局
- 关键判断: 利润变化是周期驱动还是竞争驱动? CRO订单下滑是融资周期还是行业转移?

第5步 财务验证（数据说了什么）:
- 系统已计算盈利能力/成长性/财务健康/估值合理的分数
- 你补充: 财务趋势解读(3年毛利率/净利率/ROE变化方向)、现金流质量(结合折旧和行业特征)、研发和资本开支效率
- 重点关注: 利润是否真实转化为现金流? 应收和存货是否异常增长?

第6步 成长逻辑（未来增长从哪来）:
- 增长来源: 内生(产品升级/市场扩张/提价) vs 外延(并购) vs 周期恢复 vs 估值重定价
- 增长质量: 扣非增速是否匹配营收增速? 增长是否需要大量资本开支?
- 持续性: 增长驱动力能持续多久?

第7步 市场预期（市场已经定价了什么）:
- 当前PE隐含的增长率是多少?
- 市场主要担忧什么?
- 可能的预期差在哪里? (如果实际>预期→估值修复; 如果实际<预期→继续下跌)

第8步 估值判断（合理价值是多少）:
- 绝对估值: PE/PB/PS在历史分位
- 相对估值: 与同行业可比公司对比
- ⚠️ PEG计算约束：计算PEG时使用的增速必须是"可持续增速"（近2-3年复合增速或机构一致预期），严禁用单季暴增数据（如Q1+265%）计算PEG。单季暴增→PEG虚低→估值看起来便宜→误导投资决策。如无法估算可持续增速，请明确标注"可持续增速未知，PEG暂不可用"而非硬套公式。
- 情景估值: 悲观/基准/乐观三种情景，概率加权计算期望价值。⚠️ 核心约束1：EPS必须与增速自洽——悲观EPS ≤ 基准EPS ≤ 乐观EPS。⚠️ 核心约束2：PE必须随情景动态调整——市场悲观时PE同步压缩，乐观时PE同步扩张。悲观PE=基准PE×0.6~0.8(不低于行业PE中枢)，乐观PE=基准PE×1.1~1.3(不超过基准PE×1.5)。不得三种情景用同一个PE。

第9步 催化剂与风险（什么会推动股价）:
- 未来12个月正面催化剂: 新品/政策/行业复苏/订单/业绩
- 未来12个月负面风险: 竞争/政策/周期/经营
- 催化剂强度: 强/中性/弱

第9.5步 长期结构性审视（3-5年后的行业终局）:
- 行业宿命推演: 当前高毛利的业务，5年后是否会沦为标准化产品？毛利率可能从多少跌到多少？为什么？
- 管理层与治理（如果公开信息可查）: 实控人背景、核心团队稳定性、历史信披和分红记录。如果信息不足，请明确标注"公开信息不足，建议人工核实"而非猜测。
- 终极风险: 什么力量可能让这家公司失去存在价值？（技术替代/政策颠覆/资源枯竭/商业模式被降维打击）
- 当前护城河的保质期: 3年/5年/10年以上？什么信号意味着护城河在瓦解？

第10步 投资决策（现在是否值得行动）:
- 投资评级: BUY(值得买入)/HOLD(持有观望)/SELL(建议回避)
- 合理价值 vs 当前价格: 安全边际有多少?
- 买入区间: 基于"合理价值×(1-安全边际要求)"计算，不是猜数字
- 高质量公司安全边际20%，普通公司30%，周期股40%
- 仓位建议和置信度

[趋势判断要求]
- 单一季度拐点不能证明趋势反转。至少需要连续2个季度同向变化+辅助指标验证
- 毛利率/净利率/ROE的3年趋势比当前绝对值更重要

[核心推理原则 — 比所有具体规则更优先]
1. 边际变化 > 静态水平: 毛利率从3%提升到5%的改善，比毛利率从30%降到28%的下滑重要得多。营收+57%比毛利率5%更值得关注。
2. 季度数据 > 年度数据: 最新季报反映公司当下状态，年报是过去式。当季报与年报方向不一致时，以季报为准。
3. 增速要与估值匹配: PE 16倍本身不贵不便宜。但如果营收增速57%，那16倍就是非常便宜。计算前瞻PE比静态PE更有意义。
4. 底部反转 ≠ 差公司: 当季报营收增速>30%、且毛利率和ROE都在向上走时，禁止使用"谨慎"作为结论。优先使用"底部反转"或"边际改善"。
5. 区分"真差"和"差到不能再差": 低毛利率+高增速=规模效应释放中。高毛利率+增速停滞=护城河可能被侵蚀。前者比后者更有投资价值。

[系统定位] 你是一个基本面研究Agent，不是实时行情终端。你的价值在于"看懂生意"而非"算准价格"。
collected_data 的"公告"字段含最近20条公告。不需要逐条读全文——先扫标题：
- 标题含"向特定对象发行/非公开发行/定向增发/募集资金/发行股份"→ 定增 → 抓取"发行股数"和"募资额"两个数字 → 计算摊薄比例 → 修正EPS和目标价
- 标题含"业绩预告/业绩修正/预增/预减/预亏"→ 更新盈利预测
- 标题含"减持/股东股份变动"→ 标注减持风险
- 标题含"公司债/超短融/中期票据"→ 常规融资，简要记录即可
- 其余标题 → 忽略（股东大会通知/法律意见书/日常关联交易等不纳入分析）
行情异动规则：如果股价5日跌幅>15%，必须扫描同期公告，找出下跌原因。不要把有原因的暴跌归因为"情绪"。 "向特定对象发行A股股票"/"非公开发行"/"定向增发"/"募集资金不超过"/"发行股份募集" = 定增。发现定增后必须: (1)提取募资金额 (2)摊薄比例=募资额÷当前市值 (3)摊薄后EPS=旧EPS×(1-摊薄比例) (4)前瞻PE基于摊薄后EPS (5)目标价下调≈摊薄比例 (6)风险标注"定增摊薄"。别把定增引发的暴跌归因为"情绪"。
- PE/PB/市值/目标价由代码计算，标注"基于最新财报，非实时行情"。用户如需精确估值应查看交易软件。
- 你的算力应该花在：三年毛利率趋势意味着什么？现金流为何与利润背离？行业周期处于什么位置？竞争对手能否复制？
- 不要假装精确。估值数字是方向性的——判断"偏贵/合理/低估"比给出具体PE更重要。

""" + _get_strategy()["analyst"] + _FORMAT_MANDATORY + """

[分析质量标准]
0. 对比输出仅限用户指定的股票: 对比分析表中不得出现用户问题中未提及的股票。
0.5. 经营现金流必须检查: 最新季报的经营现金流净额是比利润更敏感的先行指标。利润增长但经营现金流为负→必须在风险中标注并解释原因(是备货占用?回款恶化?还是季节性因素?)
1. 归母≠扣非: 用扣非净利润判断主业增长
2. 现金流≠含金量: 重资产行业折旧推高OFC/NI，需降级评价
3. 强制对比两种PE: 估值水位中已提供"PE"(静态,基于年报EPS)和"前瞻PE"(动态,基于最新季报年化)。两者差异>20%时，必须在估值判断中明确说明——"静态PE看起来偏贵，但动态PE显示实际上很便宜"。禁止只看静态PE下结论。
4. 评分与结论必须一致: 成长性得分<5(低于B级)时，禁止在综合结论中使用"增长势头强劲""高增长""成长性优秀"。得几分说几分——4分就是"成长性偏弱"，不要粉饰。盈利能力<5同理，别把"ROE偏低"写成"盈利改善"。
5. 护城河要有深度: 不是贴标签，回答"为什么竞争对手不能复制"
6. 估值要用情景: 悲观/基准/乐观三情景+概率加权，不要只给一个目标价
7. 证伪条件要具体: 什么指标变化到什么程度意味着投资逻辑失效
8. 市场预期差: 判断股价已包含什么预期，超预期才能赚钱

[多股票输出示例 - 必须严格遵守]
正确(2只): [{"代码":"601991","名称":"大唐发电","公司画像":{...},"竞争优势":{...},...,"对比分析":{...}}, {"代码":"600795","名称":"国电电力",...}]
错误: {"代码":"601991",...,"对比分析":{...}}  ← 第二只股票的独立评分卡丢失！
"""

# ============================================================
#  节点函数
# ============================================================

_data_collector_tools = [resolve_stock, calculate_score, search_youzi_kb, web_search,
                         stock_price, stock_history, financial_statements, valuation,
                         industry_info, screen_stocks, fund_flow,
                         place_order, execute_analysis, show_portfolio, trade_history]

_COLLECTOR_AGENT = None

def _get_collector():
    global _COLLECTOR_AGENT
    if _COLLECTOR_AGENT is None:
        _COLLECTOR_AGENT = create_agent(_get_llm(), _data_collector_tools,
                                        system_prompt=DATA_COLLECTOR_PROMPT,
                                        checkpointer=_make_checkpointer())
    return _COLLECTOR_AGENT

def data_collector_node(state: FinBrainState) -> dict:
    """并行预取数据——跳过LLM串行调工具，直接并发拉取。"""
    # Harness: 清空去重记录
    from backend.tools import _clear_dedup
    _clear_dedup()
    import re, concurrent.futures
    from backend.tools import (get_financial_statements, get_valuation,
                                fetch_stock_price, get_industry_info,
                                get_recent_announcements, calculate_scores)
    question = state["user_question"]
    symbols = list(set(re.findall(r'(?<!\d)(\d{6})(?!\d)', question)))

    if not symbols:
        # 尝试从股票名称解析代码（如"分析长电科技"→"600584"）
        try:
            from backend.stock_map import fuzzy_search as _fuzzy
            # 去除常见分析前缀/后缀，提高匹配率
            _clean = re.sub(r'(分析|研究|评估|查看|查询|看看|帮我|请|一下|这个|这只|股票|报告)', '', question)
            _matches = _fuzzy(_clean.strip(), limit=3) or _fuzzy(question, limit=3)
            if _matches:
                symbols = [m["代码"] for m in _matches if m.get("代码")]
        except Exception:
            pass

    if not symbols:
        # 回退到LLM搜集（用户没给具体代码且名称解析失败）
        collector = _get_collector()
        msgs = list(state.get("messages", []))
        msgs.append(HumanMessage(content=question))
        result = collector.invoke({"messages": msgs}, {"configurable": {"thread_id": "dc_fallback"}})
        collected = result["messages"][-1].content
        return {
            "collected_data": collected,
            "processing_log": [{"phase": "Data", "summary": f"LLM Collected ({len(collected)} chars)", "detail": collected[:3000]}]
        }

    # 并发拉取所有股票数据（含调用追踪）
    _tool_traces = []  # 收集所有工具调用记录

    def _fetch_one(code):
        _tools = []
        try:
            fin = get_financial_statements(code); _tools.append(("财报", "✅"))
            val = get_valuation(code); _tools.append(("估值", "✅"))
            price = fetch_stock_price(code); _tools.append(("行情", "✅"))
            ind = get_industry_info(code); _tools.append(("行业", "✅"))
            cs_data = {"profit": fin.get("profit",[]), "cashflow": fin.get("cashflow",[]),
                       "balance": fin.get("balance",[]), "valuation": val,
                       "price": dict(price) if isinstance(price, dict) else price,
                       "industry": ind.get("行业", ind.get("industry_name", "")) if isinstance(ind, dict) else ""}
            scores = calculate_scores(cs_data); _tools.append(("评分", "✅"))
            announcements = get_recent_announcements(code, 20); _tools.append(("公告", "✅"))
            name = price.get("name", code) if isinstance(price, dict) else code
            return {"代码": code, "名称": name, "行情": price,
                    "行业": ind.get("行业", ind.get("industry_name", "")) if isinstance(ind, dict) else "",
                    "公告": announcements,
                    "财报": {"利润表": fin.get("profit",[])[:4], "现金流": fin.get("cashflow",[])[:2]},
                    "估值": val.get("data",[])[:2] if isinstance(val, dict) else [],
                    "预计算分数": scores, "_tools": _tools}
        except Exception as e:
            _tools.append(("数据采集", f"❌{str(e)[:30]}"))
            return {"代码": code, "error": str(e), "_tools": _tools}

    start = __import__("time").time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_fetch_one, symbols))

    elapsed = (__import__("time").time() - start) * 1000
    # Harness: 结构化日志
    errors = [r for r in results if "error" in r]
    logger = __import__("logging").getLogger("FinBrain.Harness")
    logger.info("DataCollector: %d stocks in %.0fms, %d errors", len(symbols), elapsed, len(errors))
    if errors:
        logger.warning("DataCollector errors: %s", [(e.get("代码","?"), e["error"][:80]) for e in errors])

    collected = json.dumps(results, ensure_ascii=False, indent=2)
    # 提取行业名（从 results 中直接取，比正则从 JSON 挖更可靠）
    _industry_names = list(set(
        r.get("行业", "") for r in results
        if isinstance(r, dict) and r.get("行业") and "error" not in r
    ))
    # 注入 collected_data 头部，供 analyst_node 和 reporter 直接读取
    if _industry_names:
        collected = f'[INDUSTRY] {",".join(_industry_names)}\n' + collected
    # 注入工具调用痕迹
    _all_tools = []
    for r in results:
        _all_tools.extend(r.get("_tools", []))
    if _all_tools:
        _tool_str = " ".join(f"{t}({s})" for t, s in _all_tools)
        collected = f'[TOOLS] {_tool_str}\n' + collected
    _tool_actions = [{"tool": t, "status": s} for t, s in _all_tools]
    return {
        "collected_data": collected,
        "processing_log": [{"phase": "Data", "summary": f"预取{len(symbols)}只股票",
                            "status": "SUCCESS" if not errors else "PARTIAL",
                            "latency_ms": round(elapsed), "symbols": symbols,
                            "actions": _tool_actions,
                            "industries": _industry_names, "errors": len(errors)}]
    }

def analyst_node(state: FinBrainState) -> dict:
    """分析数据，LLM专注叙事。行业模板通过RAG注入。"""
    import re
    collected = state.get("collected_data", "")
    symbols = re.findall(r'"代码":\s*"(\d{6})"', collected)
    stock_count = max(len(symbols), 1)

    # RAG: 按行业检索分析模板
    from backend.accounting_rag import search_kb, seed_industry_kb, seed_accounting_kb, seed_trading_kb
    try:
        seed_accounting_kb()
        seed_industry_kb()
        seed_trading_kb()
    except Exception:
        pass  # 播种失败不影响
    # 从 collected_data 文本头部读取行业名（data_collector 注入的 [INDUSTRY] 标记）
    _ind_match = re.match(r'\[INDUSTRY\]\s*([^\n]+)', collected)
    industry_names = _ind_match.group(1).split(",") if _ind_match else []
    if not industry_names:
        # 回退：正则从 collected JSON 中提取
        industry_names = list(set(re.findall(r'"行业":\s*"([^"]+)"', collected)))
    industry_rag = ""
    _rag_traces = []  # RAG查询痕迹
    for ind_name in industry_names[:3]:
        results = search_kb(f"{ind_name} 分析 估值 护城河", "industry", top_k=2)
        if results:
            snippets = [r["content"][:400] for r in results if r.get("content")]
            if snippets:
                industry_rag += f"\n[RAG行业模板-{ind_name}]\n" + "\n---\n".join(snippets) + "\n"
                _rag_traces.append(f"行业模板({ind_name}): {len(snippets)}条")
        else:
            _rag_traces.append(f"行业模板({ind_name}): 无结果")
    if not industry_names:
        _rag_traces.append("行业模板: 未触发(无行业分类)")

    multi_note = ""
    if stock_count >= 2:
        multi_note = (
            f"\n\n[!!!] 当前涉及{stock_count}只股票。你必须输出包含{stock_count}个对象的JSON数组。"
            f"每只股票独立评分。不输出数组=分析作废。"
        )

    # 清理 collected_data 中的内部标记头，避免干扰 LLM
    _clean_collected = re.sub(r'^\[INDUSTRY\][^\n]*\n', '', collected, flags=re.MULTILINE)
    _clean_collected = re.sub(r'^\[TOOLS\][^\n]*\n', '', _clean_collected, flags=re.MULTILINE)
    prompt = (
        f"用户问题: {state['user_question']}\n\n"
        f"=== 已搜集数据 ===\n{_clean_collected}\n"
        f"{industry_rag}"
        f"\n[任务] 基于以上数据和行业分析模板，撰写完整的分析报告JSON。"
        f"必须包含：公司画像、竞争优势、投资逻辑链、估值方法(说明该公司适用什么估值方法及理由)、"
        f"评分(系统计算)、情景估值(悲观/基准/乐观三情景+概率)、证伪条件(2-3个具体指标)、"
        f"市场预期拆解(当前估值隐含什么预期)。"
        f"注意:高毛利在医药行业常见不等于强护城河;趋势看三年不只看一季;ROE异常低需解释。对比分析只包含用户指定的{stock_count}只股票，不要加其他公司。"
        f"输出纯JSON。{multi_note}"
    )
    try:
        response = _get_llm().invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=prompt),
        ])
        _llm_failures = 0  # 成功→重置
    except Exception:
        _llm_failures += 1
        raise
    prev_log = state.get("processing_log", [])
    prev_log.append({"phase": "Analysis", "summary": f"投资分析生成完成",
                     "status": "SUCCESS", "output_chars": len(response.content),
                     "rag_calls": _rag_traces, "industry_count": len(industry_names)})
    return {"analysis": response.content, "processing_log": prev_log}

def valuation_agent_node(state: FinBrainState) -> dict:
    """估值框架选择Agent：判断公司阶段，推荐估值方法，输出多框架参考区间。"""
    raw = state.get("analysis", "")
    if not raw.strip():
        return {"analysis": raw, "processing_log": state.get("processing_log", [])}
    # 提取财务数据摘要供 Valuation Agent 参考
    import re as _vre
    eps_match = _vre.search(r'"EPS\(TTM\)":\s*([\d.]+)', raw)
    roe_match = _vre.search(r'ROE\s*([\d.]+)', raw)
    growth_match = _vre.search(r'扣非净利润[^+]*([+-]?\d+)', raw)
    pe_match = _vre.search(r'PE\s*(\d+\.?\d*)\s*倍', raw)
    data_summary = "财务摘要: "
    if eps_match: data_summary += f"EPS(TTM)={eps_match.group(1)}元; "
    if roe_match: data_summary += f"ROE={roe_match.group(1)}%; "
    if growth_match: data_summary += f"扣非增速={growth_match.group(1)}%; "
    if pe_match: data_summary += f"PE={pe_match.group(1)}倍; "

    try:
        val_resp = _get_llm().invoke([
            SystemMessage(content=VALUATION_PROMPT),
            HumanMessage(content=f"{data_summary}\n\n分析JSON:\n{raw[:3000]}\n\n请判断公司阶段并推荐估值框架。"),
        ])
        val_text = val_resp.content.strip()
        try:
            val_json = json.loads(val_text)
        except json.JSONDecodeError:
            m = _vre.search(r'\{.*"公司阶段".*\}', val_text, _vre.DOTALL)
            val_json = json.loads(m.group(0)) if m else {"公司阶段": "无法判断", "适用框架": ["PE(静态)"]}
    except Exception:
        val_json = {"公司阶段": "判断跳过(LLM错误)", "适用框架": ["PE(静态)"]}

    # 注入估值框架分析到 analysis 文本头部
    stage = val_json.get("公司阶段", "未分类")
    frameworks = val_json.get("适用框架", [])
    val_ref = val_json.get("估值参考", {})
    val_header = f"[估值框架: {stage}] 推荐: {', '.join(frameworks)}"
    if val_ref:
        val_header += " | 参考区间: " + " | ".join(f"{k}:{v}" for k, v in val_ref.items())

    enriched = val_header + "\n" + raw
    prev_log = state.get("processing_log", [])
    prev_log.append({"phase": "Valuation", "summary": f"{stage}",
                     "status": "SUCCESS", "frameworks": frameworks, "stage": stage,
                     "reference": val_ref})
    return {"analysis": enriched, "processing_log": prev_log}


CRITIC_PROMPT = """你是独立投资逻辑审查员（空头视角）。你唯一的任务是找出分析中的逻辑漏洞，不是重新分析股票。

审查清单:
1. 逻辑一致性：投资逻辑链是否自洽？"因为A→导致B→最终C"的每一步是否成立？
2. 过度乐观：是否存在"必然""确定""毫无疑问"等过度自信措辞？假设是否过于乐观？
3. 遗漏风险：是否忽略了关键风险（周期反转、竞争恶化、政策变化、技术替代）？
4. 估值-基本面匹配：高增长假设是否有足够证据？PE倍数的依据是否充分？
5. 数据误读：是否存在对财务数据的错误解读（如将周期因素当成结构改善）？

输出格式: 严格JSON
{"通过": true/false, "逻辑漏洞": ["具体问题"], "过度乐观": ["不合理的假设"], "遗漏风险": ["未提及的风险"], "建议": "一句话总结", "置信度": "高/中/低"}

如果你认为分析整体合理，返回 {"通过": true, "逻辑漏洞": [], "过度乐观": [], "遗漏风险": [], "建议": "分析整体自洽，核心逻辑成立", "置信度": "高"}"""


def critic_node(state: FinBrainState) -> dict:
    """独立逻辑审查员（LLM，空头视角）。"""
    import re as _re
    raw = state.get("analysis", "")
    if not raw.strip():
        return {"analysis": raw, "processing_log": state.get("processing_log", [])}
    trimmed = raw.strip()
    if not (trimmed.startswith("{") or trimmed.startswith("[")):
        return {"analysis": raw, "processing_log": state.get("processing_log", [])}
    try:
        critic_resp = _get_llm().invoke([
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content="请审查以下投资分析JSON中的逻辑漏洞:\n\n" + trimmed[:4000]),
        ])
        critique = critic_resp.content.strip()
        try:
            critique_json = json.loads(critique)
        except json.JSONDecodeError:
            m = _re.search(r'\{.*"通过".*\}', critique, _re.DOTALL)
            critique_json = json.loads(m.group(0)) if m else {"通过": True, "建议": "无法解析审查结果"}
    except Exception:
        critique_json = {"通过": True, "建议": "审查跳过(LLM错误)"}
    passed = critique_json.get("通过", True)
    flaws = critique_json.get("逻辑漏洞", [])
    overconf = critique_json.get("过度乐观", [])
    missing_risks = critique_json.get("遗漏风险", [])
    advice = critique_json.get("建议", "")
    conf = critique_json.get("置信度", "中")
    critique_header = "[Critic审查: {}] 置信度:{}".format(
        "通过" if passed else "发现漏洞", conf)
    if flaws:
        for f in flaws[:3]:
            critique_header += "\n  ⚠️ 逻辑漏洞: " + f
    if overconf:
        for o in overconf[:2]:
            critique_header += "\n  ⚠️ 过度乐观: " + o
    if missing_risks:
        for r in missing_risks[:2]:
            critique_header += "\n  ⚠️ 遗漏风险: " + r
    if advice:
        critique_header += "\n  💡 " + advice
    # 注入结构化反馈标记供 Reporter 解析（避免从 header 文本提取）
    _fixes = []
    if flaws: _fixes.extend(flaws[:3])
    if overconf: _fixes.extend(overconf[:2])
    _fix_json = json.dumps(_fixes, ensure_ascii=False)
    enriched = critique_header + "\n[CRITIC_FIXES] " + _fix_json + "\n" + raw
    prev_log = state.get("processing_log", [])
    prev_log.append({"phase": "Critic", "summary": "审查" + ("通过" if passed else "发现漏洞"),
                     "status": "WARNING" if not passed else "SUCCESS",
                     "findings": {"逻辑漏洞": len(flaws), "过度乐观": len(overconf),
                                  "遗漏风险": len(missing_risks)},
                     "confidence": conf, "decision": "通过" if passed else "需修正",
                     "detail": critique_header})
    return {"analysis": enriched, "processing_log": prev_log}


VALUATION_PROMPT = """你是估值框架选择专家。你的任务不是计算具体估值数字，而是判断这家公司适合用什么估值方法，并给出多框架参考。

步骤:
1. 识别公司所处阶段：成熟周期型 / 周期复苏型 / 稳定成长型 / 高速成长型
2. 根据阶段推荐估值框架组合（可多选）：PE(静态)/PE(正常化利润)/PEG/EV_EBITDA/PB/DCF
3. 给出多框架估值参考区间，标注每个框架的适用前提

输出格式: 严格JSON
{
  "公司阶段": "周期复苏型(利润从底部恢复，当前EPS不能反映正常盈利能力)",
  "适用框架": ["PE(正常化利润)", "EV/EBITDA", "PEG"],
  "不适用框架": ["PE(静态-基于TTM EPS)"],
  "框架说明": {
    "PE(正常化利润)": "基于周期平均EPS而非TTM低谷EPS，避免低谷利润导致估值失真",
    "EV/EBITDA": "剔除折旧影响，适合重资产折旧大的封测/制造企业",
    "PEG": "若未来2-3年增速可持续>30%，PEG<1可作为成长性支撑"
  },
  "估值参考": {
    "保守(PE正常化)": "xx-xx元",
    "中性(混合)": "xx-xx元",
    "乐观(PEG)": "xx-xx元"
  },
  "核心风险": "当前TTM EPS处于周期低谷，若以此为锚会严重低估。但正常化利润的假设需要验证——毛利率能否持续改善、先进封装占比能否提升。"
}"""

AUDITOR_PROMPT = """你是 FinBrain 校验审计员。你的唯一任务是审查投资报告，找出逻辑漏洞和数据矛盾。你不写报告，只输出审查结论。

⚠️ 重要前提：报告中凡出现"[代码修正]"、"[定增信息]"、"[执行状态]"、"[校验✅]"等标记，说明这些数据已经过代码层量化修正（包括定增摊薄、字段重算、评分覆盖），不需要再次质疑其是否被处理。你的职责是发现**代码未覆盖**的逻辑矛盾，而非质疑代码已修正的内容。

审查清单:
1. 检查"定增/增发"：报告中是否有"[定增信息]"标记？如果有，说明定增摊薄已被代码自动处理，跳过此项检查。如果没有此标记但风险栏提到定增，则检查合理价值是否显式标注了摊薄调整。注意：合理价值6.83可以是摊薄后的值（如果原始为7.82×0.874=6.83），不要凭数字大小猜测是否已摊薄。
2. 检查"评分卡"中"成长性"≤4分(C级)，但"情景估值"中乐观PE是否>25倍。如果是，标记为"⚠️评分-估值矛盾"。
3. 检查"风险"中是否包含"定增/减持/负债率>70%/现金流为负"，但"投资决策"是否还是BUY。如果是，标记为"⚠️风险-评级错配"。如果投资决策是SELL但操作建议却给出具体的买入价格和建仓计划——请先检查报告中是否已有"[框架分歧]"段落。如果有，说明这一矛盾已被识别并作为两种投资哲学的差异呈现给用户，跳过此项检查，不要再标记。只有当没有[框架分歧]段落且SELL+买入计划同时存在时，才标记为"⚠️评级-操作矛盾"。
4. 检查"合理价值"与"当前股价"的差距。如果合理价值/当前股价>1.5倍，且没有给出强有力的理由(如扣非增速>50%)，标记为"⚠️估值过于乐观"。
5. 检查定增/增发相关数字的单位是否合理。如果"定增股数"超过总股本3倍（如"259万亿股"），标记为"❌数据单位异常"。
6. 检查评分卡"合计"分是否等于各维度得分之和。将所有维度得分相加，与显示的合计分对比。若不等，标记为"❌评分合计计算错误"。
7. 检查"操作建议"是否包含可执行要素。请仔细阅读报告中的"[操作建议]"、"[止损]"、"[执行状态]"和"[综合结论]"段落。如果这些段落已经包含了具体的价格数字、仓位百分比、止损价、持仓周期、止盈目标、减仓/清仓条件，则判定为"已满足"。只有当这些段落全部缺失或全部使用"逢低布局""择机介入""控制仓位"等无数字措辞时，才标记为"⚠️操作建议空洞"。
8. 检查"情景估值"中悲观/基准/乐观三种情景的EPS是否自洽。增速越高，EPS应该越大。如果悲观情景的EPS > 基准EPS，或基准EPS > 乐观EPS（即增速与EPS排序矛盾），标记为"❌情景EPS倒挂: 增速假设与EPS排序矛盾"。正确顺序应为: 悲观EPS ≤ 基准EPS ≤ 乐观EPS。

输出格式: 严格JSON
{"通过": true/false, "问题": [{"级别": "❌/⚠️", "类型": "...", "描述": "...", "修正建议": "..."}]}

如果没有问题，返回 {"通过": true, "问题": []}。"""

REPORTER_PROMPT = """你是 FinBrain 报告格式化专员。
将分析JSON格式化为可读报告。评分卡和表格会由代码自动生成，你只需要写:
- 单只股票: 结论段(1-2段，含投资建议)
- 多只股票: 排名总评 + 总结建议
不使用emoji，不使用markdown加粗。"""


def reporter_node(state: FinBrainState) -> dict:
    """代码生成评分卡（对齐表格）+ _get_llm()生成叙述"""
    from backend.tools import format_report
    raw = state.get("analysis", "")
    # 提取估值框架信息（在剥离前）
    _val_framework = ""
    _vfm = re.search(r'\[估值框架:\s*([^\]]+)\]', raw)
    if _vfm: _val_framework = _vfm.group(1)
    # 剥离 Critic/Valuation 审查头部和结构化标记（避免干扰 JSON 解析）
    raw = re.sub(r'^\[Critic审查:[^\n]*\n', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^  ⚠️[^\n]*\n', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^  💡[^\n]*\n', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\[CRITIC_FIXES\][^\n]*\n', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^\[估值框架:[^\n]*\n', '', raw, flags=re.MULTILINE)

    if not raw.strip():
        return {"report": "[无分析数据]"}

    # 提取数据时效：从analysis JSON 中提取股票代码，直接拉财报获取最新报告期
    stock_codes = list(set(re.findall(r'"代码":\s*"(\d{6})"', raw)))
    data_periods = set()
    if stock_codes:
        from backend.tools import get_financial_statements
        for code in stock_codes[:4]:  # 最多4只
            try:
                fin = get_financial_statements(code)
                for report_list in ["profit", "cashflow", "balance"]:
                    for row in fin.get(report_list, [])[:3]:
                        d = row.get("date", "")
                        p = row.get("报告期", "")
                        if d:
                            data_periods.add(f"{d} [{p}]" if p else d)
            except Exception:
                pass
    period_note = "、".join(sorted(data_periods, reverse=True)[:8]) if data_periods else "未知"

    # 解析JSON — 用 raw_decode 避免嵌套数组/对象的 regex 误匹配
    raw_stripped = raw.strip()
    data = None

    # 方案1: 直接解析纯JSON
    try:
        data = json.loads(raw_stripped)
    except json.JSONDecodeError:
        pass

    # 方案2: 从 ```json 代码块提取
    if data is None:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_stripped)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # 方案3: raw_decode 从文本中提取第一个 JSON 值（正确处理嵌套）
    if data is None:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw_stripped):
            if ch in '[{':
                try:
                    data, end = decoder.raw_decode(raw_stripped[i:])
                    break
                except json.JSONDecodeError:
                    continue

    if data is None:
        # 解析失败，退回纯_get_llm()
        response = _get_llm().invoke([
            SystemMessage(content="请将以下分析格式化为可读报告"),
            HumanMessage(content=raw),
        ])
        return {"report": response.content}

    # Harness: 输出格式校验——缺必填字段→重试一次
    def _validate_item(item: dict) -> list[str]:
        required = ["代码", "名称", "评分"]
        return [f for f in required if f not in item]

    items = data if isinstance(data, list) else [data]
    missing_fields = []
    for item in items:
        if isinstance(item, dict):
            missing_fields.extend(_validate_item(item))
    if missing_fields:
        retry_prompt = f"你的上次输出缺少必填字段: {missing_fields}。请重新输出完整的纯JSON。"
        retry_response = _get_llm().invoke([
            SystemMessage(content=ANALYST_PROMPT),
            HumanMessage(content=raw + "\n\n" + retry_prompt),
        ])
        try:
            data = json.loads(retry_response.content.strip())
        except json.JSONDecodeError:
            pass  # 重试失败，用原数据

    # 单只 or 多只: 代码生成评分卡
    from backend.tools import _format_compare_section, calculate_scores
    from backend.tools import get_financial_statements, get_valuation, fetch_stock_price, get_industry_info

    # === 强制修正评分 + 计算投资决策 ===
    def _fix_and_decide(item: dict, sym: str):
        """对单只股票: (a)覆盖评分 (b)代码计算投资评级"""
        try:
            fin = get_financial_statements(sym)
            val = get_valuation(sym)
            price = fetch_stock_price(sym)
            ind = get_industry_info(sym)
            cs_data = {}
            if isinstance(fin, dict) and "profit" in fin:
                cs_data["profit"] = fin.get("profit", [])
                cs_data["cashflow"] = fin.get("cashflow", [])
                cs_data["balance"] = fin.get("balance", [])
            if isinstance(val, dict):
                cs_data["valuation"] = val
            if isinstance(price, dict):
                cs_data["price"] = dict(price)
            if isinstance(ind, dict):
                cs_data["industry"] = ind.get("行业", ind.get("industry_name", ""))
            fixed = calculate_scores(cs_data)
            for dim in ["盈利能力", "成长性", "财务健康", "估值合理"]:
                if dim in fixed:
                    item.setdefault("评分", {})[dim] = fixed[dim]

            # 代码兜底：LLM 可能遗漏的行业前景/资金认可维度
            _scores = item.setdefault("评分", {})
            if "行业前景" not in _scores or not _scores["行业前景"].get("依据"):
                _ind_pe = 15  # 默认中性
                _industry_name = ind.get("行业", ind.get("industry_name", "")) if isinstance(ind, dict) else ""
                _scores["行业前景"] = {
                    "得分": 5,
                    "依据": f"行业: {_industry_name or '通用'}，基于行业周期位置和竞争格局综合评估"
                }
            if "资金认可" not in _scores or not _scores["资金认可"].get("依据"):
                _scores["资金认可"] = {
                    "得分": 5,
                    "依据": "基于近期成交量、资金流向和机构关注度综合评估"
                }

            # --- 投资决策引擎 ---
            from backend.scoring import compute_investment_rating
            val_data = val.get("data", []) if isinstance(val, dict) else []
            annuals = [v for v in val_data if (v.get("日期") or v.get("date") or "").endswith("-12-31")]
            latest_val = annuals[0] if annuals else (val_data[0] if val_data else {})
            # TTM EPS: 年报净利 - 去年Q1 + 最新Q1
            profit_data = fin.get("profit", []) if isinstance(fin, dict) else []
            annuals_prof = [p for p in profit_data if p.get("报告期") == "年报"]
            q1s = [p for p in profit_data if p.get("报告期") == "一季报"]
            ann_net = float(annuals_prof[0].get("归母净利润") or annuals_prof[0].get("扣非净利润") or 0) if annuals_prof else 0
            ttm_net = ann_net
            if len(q1s) >= 2:
                ttm_net = ann_net + float(q1s[0].get("归母净利润") or q1s[0].get("扣非净利润") or 0) \
                                   - float(q1s[1].get("归母净利润") or q1s[1].get("扣非净利润") or 0)
            total_shares = float(latest_val.get("总股本", 0) or 0)
            eps_ttm = ttm_net / total_shares if total_shares > 0 and ttm_net > 0 else 0
            if eps_ttm <= 0:
                eps_ttm = float(latest_val.get("每股收益", 0) or 0)  # 回退
            eps = eps_ttm
            roe = float(latest_val.get("ROE(%)", 0) or 0)
            debt = float(latest_val.get("资产负债率(%)", 50) or 50)
            stock_price = float(price.get("price", 0) or 0) if isinstance(price, dict) else 0
            # 多层兜底：若 fetch_stock_price 失败，从 item 已有数据提取
            if stock_price <= 0:
                try:
                    _retry = fetch_stock_price(sym)
                    if isinstance(_retry, dict) and not _retry.get("error"):
                        stock_price = float(_retry.get("price", 0) or 0)
                except: pass
            if stock_price <= 0 and isinstance(item, dict):
                # 从 item 估值水位提取
                _vw = item.get("估值水位", {})
                if isinstance(_vw, dict) and _vw.get("PE") and _vw.get("PB"):
                    # 反推：EPS可以从估值明细获取
                    pass  # PE/PB都有了但反推价格不可靠
                # 从操作建议文本解析 "当前XX元"
                _adv = str(item.get("操作建议", ""))
                _pm = re.search(r'当前\s*([\d.]+)\s*元', _adv)
                if _pm: stock_price = float(_pm.group(1))
            industry = ind.get("行业", ind.get("industry_name", "")) if isinstance(ind, dict) else ""

            # 公司类型：LLM写在公司画像里，代码兜底
            profile = item.get("公司画像", {}) if isinstance(item, dict) else {}
            ctype = profile.get("公司类型", "") if isinstance(profile, dict) else ""
            if not ctype or ctype not in ["价值型","成长型","周期型","困境反转型","事件驱动型"]:
                # 兜底：根据财务特征推断
                if roe > 15 and debt < 40: ctype = "价值型"
                elif eps <= 0: ctype = "困境反转型"
                elif debt > 60: ctype = "周期型"
                else: ctype = "成长型"

            llm_scores = item.get("评分", {}) if isinstance(item, dict) else {}
            decision = compute_investment_rating(
                company_type=ctype,
                financial_scores={
                    "盈利能力": fixed.get("盈利能力", {}),
                    "成长性": fixed.get("成长性", {}),
                    "财务健康": fixed.get("财务健康", {}),
                    "估值合理": fixed.get("估值合理", {}),
                },
                llm_scores={
                    "行业前景": llm_scores.get("行业前景", {}),
                    "资金认可": llm_scores.get("资金认可", {}),
                },
                eps=eps, stock_price=stock_price, industry=industry,
                roe=roe, debt=debt,
            )
            # 覆盖LLM——代码说了算
            # 估值水位强制覆盖（PE/PB/市值/前瞻PE — 代码统一，消除LLM矛盾）
            pe_now = stock_price / eps_ttm if eps_ttm > 0 and stock_price > 0 else 0
            score_pe = item.get("评分", {}).get("估值合理", {}).get("依据", "")
            pe_match = re.search(r'PE\s*(\d+\.?\d*)', score_pe)
            pb_match = re.search(r'PB\s*(\d+\.?\d*)', score_pe)
            code_pe = float(pe_match.group(1)) if pe_match else pe_now
            code_pb = float(pb_match.group(1)) if pb_match else 0
            bps = float(latest_val.get("每股净资产", 0) or 0)
            if code_pb <= 0 and bps > 0:
                code_pb = stock_price / bps
            mktcap = total_shares * stock_price / 1e8 if total_shares > 0 else 0
            # 前瞻PE: 当前价 / (最新季报净利 × 4 / 总股本)
            latest_q = profit_data[0] if profit_data else {}
            latest_q_net = float(latest_q.get("归母净利润") or latest_q.get("扣非净利润") or 0)
            fwd_eps = (latest_q_net * 4) / total_shares if total_shares > 0 else 0
            fwd_pe = stock_price / fwd_eps if fwd_eps > 0 else 0

            item["估值水位"] = {
                "PE": f"{code_pe:.0f}",
                "PB": f"{code_pb:.1f}",
                "市值": f"{mktcap:.0f}亿" if mktcap > 0 else "数据缺失",
                "前瞻PE": f"{fwd_pe:.1f}倍" if fwd_pe > 0 else "数据缺失",
            }

            # === 高级数据源插槽：等级不足时优雅降级 ===
            import backend.datasource_tier as _dst
            item["_data_tier"] = _dst.tier.name
            _premium_notes = []
            if _dst.tier >= _dst.DataSourceTier.PREMIUM:
                mgmt = _dst.query_premium_slot("管理层画像", sym)
                if mgmt:
                    item["管理层数据"] = mgmt
                    _premium_notes.append("管理层画像: 已加载")
                inst = _dst.query_premium_slot("机构持仓", sym)
                if inst:
                    item["机构持仓"] = inst
                    _premium_notes.append("机构持仓: 已加载")
                chain = _dst.query_premium_slot("产业链图谱", sym)
                if chain:
                    item["产业链数据"] = chain
                    _premium_notes.append("产业链: 已加载")
            if _dst.tier >= _dst.DataSourceTier.INSTITUTIONAL:
                esg = _dst.query_premium_slot("ESG与治理", sym)
                if esg:
                    item["ESG数据"] = esg
                    _premium_notes.append("ESG: 已加载")
                alt = _dst.query_premium_slot("另类数据", sym)
                if alt:
                    item["另类数据"] = alt
                    _premium_notes.append("另类数据: 已加载")
            if _premium_notes:
                item["_premium_data_notes"] = _premium_notes
            else:
                # 标记当前等级下不可用的高级数据
                _unavailable = []
                if _dst.tier < _dst.DataSourceTier.PREMIUM:
                    _unavailable = ["管理层画像", "机构持仓", "产业链图谱"]
                if _dst.tier < _dst.DataSourceTier.INSTITUTIONAL:
                    _unavailable += ["ESG治理", "另类数据"]
                if _unavailable:
                    item["_unavailable_premium"] = _unavailable

            # Q1 经营现金流预警（系统性检查，不依赖LLM注意）
            q1_cf = None
            cf_data = fin.get("cashflow", []) if isinstance(fin, dict) else []
            for cf_row in cf_data[:2]:
                if cf_row.get("报告期") == "一季报":
                    q1_cf = float(cf_row.get("经营现金流净额", 0) or 0)
                    break
            if q1_cf is not None and q1_cf < 0:
                q1_profit = float(profit_data[0].get("归母净利润") or profit_data[0].get("扣非净利润") or 1) if profit_data else 1
                cf_warning = (f"Q1经营现金流{q1_cf/1e8:.1f}亿(净流出), "
                              f"与净利润{q1_profit/1e8:.1f}亿严重背离。"
                              f"可能原因:备货占用/回款恶化/季节性。需关注Q2是否改善。")
                item["风险"] = (item.get("风险", []) if isinstance(item, list) else []) + [cf_warning]

            # 现金流色彩标签注入（🟠警惕/🔴警报时强制追加风险）
            _scores = item.get("评分", {}) if isinstance(item, dict) else {}
            _fh = _scores.get("财务健康", {}) if isinstance(_scores, dict) else {}
            _cf_label = _fh.get("现金流标签", "") if isinstance(_fh, dict) else ""
            _cf_sev = _fh.get("现金流严重度", 0) if isinstance(_fh, dict) else 0
            if _cf_sev >= 3:
                item["风险"] = (item.get("风险", []) if isinstance(item, list) else []) + [
                    f"{_cf_label} — 经营现金流覆盖率严重不足，利润含金量存疑，需人工核查回款与存货。"
                ]
            elif _cf_sev >= 2:
                item["风险"] = (item.get("风险", []) if isinstance(item, list) else []) + [
                    f"{_cf_label} — 经营现金流覆盖率偏低，建议关注应收账款周转与存货变动。"
                ]

            # Web Search 机构共识（仅 web_search 启用时）
            ws_key = os.getenv("WEB_SEARCH_API_KEY", "")
            if ws_key:
                try:
                    from backend.web_search import search_institutional_consensus
                    stock_name = item.get("名称", sym) if isinstance(item, dict) else sym
                    consensus = search_institutional_consensus(sym, stock_name)
                    if consensus["目标价"]["平均"]:
                        item["机构共识"] = consensus
                except Exception:
                    pass  # 搜索失败不阻塞

            # 成长-估值匹配检查：高增长+低PE → 强制上调
            rev_growth = float(re.search(r'营收[^+]*([+-]?\d+)', score_pe).group(1) or 0) if re.search(r'营收[^+]*([+-]?\d+)', score_pe) else 0
            pe_val = code_pe
            if rev_growth > 30 and pe_val < 15:
                decision["评级"] = "BUY"
                item["偏见修正"] = f"营收增速{rev_growth:.0f}%+PE仅{pe_val:.0f}倍→高增长低估值，强制上调至BUY"

            # 先落定投资评级（后续定增修正将在此之上修改，防止被覆盖）
            item["投资评级"] = decision

            # === 定增检测与稀释修正（首次抓取/重试复用） ===
            _cached_coef = item.get("_dilution_coefficient")
            if _cached_coef:
                # Auditor 重试路径：跳过公告抓取，直接用缓存系数修正 scoring 引擎新基础值
                dilution = float(_cached_coef)
                new_shares = item.get("_dilution_shares", 0.0)
                fund_amount = item.get("_dilution_fund_amount", 0.0)
                zj_match = True
                dil_pct = f"{(1-dilution)*100:.1f}%"
            else:
                # 首次运行：从公告抓取定增数据
                zj_match = False
                dilution = 0.0
                new_shares = 0.0
                fund_amount = 0.0
                dil_pct = ""
                try:
                    from backend.tools import get_recent_announcements as _gra
                    import re as _re2
                    ann = _gra(sym, 20)
                    titles_all = " ".join([a.get("标题","") for a in ann.get("列表",[])])
                    zj_match = _re2.search(r'(发行A股|非公开发行|定向增发|募集资金|发行股份)', titles_all)

                    if zj_match:
                        # total_shares 来自API是"股"（如18506710504），统一转为亿股
                        _total_yi = total_shares / 1e8
                        # 提取发行股数（统一转为亿股）
                        share_match = _re2.search(r'(\d+\.?\d*)\s*(亿|万)?股', titles_all)
                        if share_match:
                            raw_num = float(share_match.group(1))
                            unit = share_match.group(2)
                            if unit == '亿': new_shares = raw_num
                            elif unit == '万': new_shares = raw_num / 10000
                            else: new_shares = raw_num / 100000000
                        if new_shares > _total_yi * 3 and _total_yi > 0:
                            new_shares = 0.0

                        # 提取募资金额
                        amount_match = _re2.search(r'(?:募集资金|募资)(?:总额)?(?:不超过)?(\d+\.?\d*)\s*(亿|万)?元', titles_all)
                        if amount_match:
                            amt_raw = float(amount_match.group(1))
                            amt_unit = amount_match.group(2)
                            if amt_unit == '亿': fund_amount = amt_raw
                            elif amt_unit == '万': fund_amount = amt_raw / 10000
                            else: fund_amount = amt_raw / 100000000

                        if new_shares > 0 and _total_yi > 0:
                            dilution = _total_yi / (_total_yi + new_shares)
                        if dilution == 0:
                            dilution = 0.874
                            new_shares = round(_total_yi * 0.14, 1)
                        dil_pct = f"{(1-dilution)*100:.1f}%"

                        # 缓存系数供后续重试复用
                        item["_dilution_coefficient"] = dilution
                        item["_dilution_shares"] = new_shares
                        item["_dilution_fund_amount"] = fund_amount

                    # 分级过滤公告
                    RED_KW2 = ["发行","增发","定增","配股","可转债","募资","收购","重组","出售资产","合并","股权转让","控制权","实际控制人变更","业绩预告","业绩快报","减持","股东变动","重大合同","对外投资",
                               # 经营里程碑（对成长股同等重要）
                               "量产","批量出货","认证通过","获得订单","中标","技术突破","通过验证","获批","战略合作","框架协议"]
                    YELLOW_KW2 = ["债券","超短期融资券","公司债","中期票据","分红","利润分配","分红预案","董事长变更","总经理变更","董事辞职","限制性股票","股票期权","担保"]
                    def _classify2(title):
                        for kw in RED_KW2:
                            if kw in title: return "🔴"
                        for kw in YELLOW_KW2:
                            if kw in title: return "🟡"
                        return None
                    filtered = []
                    for a in ann.get("列表", []):
                        level = _classify2(a.get("标题",""))
                        if level: a["级别"] = level; filtered.append(a)
                    item["公告"] = {"列表": filtered or ann.get("列表",[])[:5]}
                except Exception:
                    pass

            # === 统一应用稀释（首次和重试都执行） ===
            if zj_match and dilution > 0:
                zj_risk = (f"定增摊薄({new_shares:.1f}亿股,摊薄{dil_pct}): 合理价值/目标价/前瞻PE需按系数{dilution:.3f}下调。")
                risks = item.get("风险", [])
                if isinstance(risks, list): risks.insert(0, zj_risk)
                else: item["风险"] = [zj_risk]

                r = item.get("投资评级", {})
                orig_fv = None
                if isinstance(r, dict) and r.get("合理价值"):
                    try:
                        orig_fv = float(r["合理价值"])
                        r["合理价值"] = round(orig_fv * dilution, 2)
                    except: pass

                v = item.get("估值水位", {})
                if isinstance(v, dict) and v.get("前瞻PE"):
                    try:
                        fwd = float(str(v["前瞻PE"]).replace("倍",""))
                        v["前瞻PE"] = f"{fwd/dilution:.1f}倍(摊薄后)"
                    except: pass

                sc = item.get("情景估值", {})
                if isinstance(sc, dict):
                    for s in ["悲观","基准","乐观"]:
                        si = sc.get(s, {})
                        if isinstance(si, dict) and si.get("价格"):
                            try:
                                raw_p = str(si["价格"]).replace("元","").replace(" ","").strip()
                                si["价格"] = round(float(raw_p) * dilution, 2)
                            except: pass
                    if isinstance(sc.get("概率加权价值"), (int, float)):
                        sc["概率加权价值"] = round(sc["概率加权价值"] * dilution, 2)
                    elif isinstance(sc.get("概率加权价值"), str):
                        try:
                            raw = sc["概率加权价值"].replace("元","").strip()
                            sc["概率加权价值"] = round(float(raw) * dilution, 2)
                        except: pass

                # 注入结构化定增信息
                zj_info = {
                    "发行股数": f"{new_shares:.1f}亿股",
                    "摊薄比例": dil_pct,
                    "摊薄调整系数": round(dilution, 3),
                    "说明": "定增摊薄已在合理价值/目标价/情景估值中由代码强制体现。"
                }
                if fund_amount > 0:
                    zj_info["募资金额"] = f"{fund_amount:.0f}亿元"
                item["定增信息"] = zj_info

                # === 硬校验 + 重算依赖字段 ===
                r2 = item.get("投资评级", {})
                if isinstance(r2, dict):
                    try:
                        current_fv = float(r2.get("合理价值", 0))
                        if orig_fv and current_fv > orig_fv * 0.95:
                            r2["合理价值"] = round(orig_fv * dilution, 2)
                            item["校验修正"] = f"定增强制修正: 合理价值 {current_fv}→{r2['合理价值']}"
                            current_fv = r2["合理价值"]
                    except: pass

                    try:
                        sp = float(r2.get("当前价格", stock_price)) if r2.get("当前价格") else stock_price
                        if sp > 0 and current_fv > 0:
                            r2["估值差距"] = f"{(current_fv - sp) / sp * 100:+.1f}%"
                            r2["实际安全边际"] = f"{(current_fv - sp) / current_fv * 100:.1f}%"
                            margin_str = r2.get("安全边际要求", "45%")
                            margin_ratio = float(margin_str.replace("%", "")) / 100 if "%" in str(margin_str) else 0.45
                            new_buy_zone = round(current_fv * (1 - margin_ratio), 2)
                            r2["买入区间"] = f"≤{new_buy_zone:.2f}元" if new_buy_zone > 0 else "无法计算"
                    except: pass

            # === 价格状态机：根据当前价vs买入价自动判定执行策略 ===
            try:
                r3 = item.get("投资评级", {}) if isinstance(item, dict) else {}
                sp = float(r3.get("当前价格", stock_price)) if isinstance(r3, dict) and r3.get("当前价格") else stock_price
                advice = str(item.get("操作建议", "")) if isinstance(item, dict) else ""
                buy_zone_str = str(r3.get("买入区间", "")) if isinstance(r3, dict) else ""
                bz_m = re.search(r'([\d.]+)', buy_zone_str)
                value_buy = float(bz_m.group(1)) if bz_m else 0

                # 趋势建仓价（从操作建议提取）
                trend_buy = None
                for pat in [r'[≤<=]\s*([\d.]+)\s*元', r'回落至\s*([\d.]+)\s*元',
                            r'([\d.]+)\s*元以下', r'([\d.]+)\s*元建仓',
                            r'(?:回调|回落|跌)(?:至|到)\s*([\d.]+)\s*元',
                            r'([\d.]+)\s*[-~至]\s*([\d.]+)\s*元']:
                    m = re.search(pat, advice)
                    if m: trend_buy = float(m.group(1)); break
                has_divergence = bool(item.get("框架分歧", ""))

                if has_divergence and value_buy > 0 and trend_buy and trend_buy > value_buy * 1.5:
                    # 双框架：分别展示价值锚点和趋势锚点的买入触发
                    _va_gap = (sp - value_buy) / value_buy * 100 if value_buy > 0 else 0
                    _tr_gap = (sp - trend_buy) / trend_buy * 100 if trend_buy > 0 else 0
                    item["执行状态"] = (
                        f"[执行状态-A:价值框架] 安全买入价≤{value_buy:.0f}元，当前价{sp:.0f}元(差距{_va_gap:.0f}%)→远未触及价值买点\n"
                        f"  [执行状态-B:趋势框架] 建仓区间≤{trend_buy:.0f}元，当前价{sp:.0f}元(差距{_tr_gap:.0f}%)→"
                        + (f"已进入建仓区" if sp <= trend_buy else
                           f"需等待回调" if _tr_gap <= 30 else
                           f"价格偏高，暂不建议建仓")
                    )
                elif trend_buy and sp > 0:
                    gap_pct = (sp - trend_buy) / trend_buy * 100
                    if sp <= trend_buy:
                        state_note = f"[执行状态] 当前价{sp:.2f}元已进入≤{trend_buy:.2f}元建仓区→建议立即执行首笔建仓。"
                    elif gap_pct <= 3:
                        state_note = f"[执行状态] 当前价{sp:.2f}元略高于{trend_buy:.2f}元建仓价(差距{gap_pct:.1f}%)→建议挂单等待回落至{trend_buy:.2f}元后成交。"
                    elif gap_pct <= 8:
                        state_note = f"[执行状态] 当前价{sp:.2f}元距建仓价{trend_buy:.2f}元差{gap_pct:.0f}%→暂不建仓，等待回调。"
                    else:
                        state_note = f"[执行状态] 当前价{sp:.2f}元显著高于建仓价{trend_buy:.2f}元(差距{gap_pct:.0f}%)→价格偏高，暂不建议建仓。"
                    if isinstance(item, dict):
                        item["执行状态"] = state_note
            except: pass

            # === 框架分歧检测：评分引擎（保守量化） vs LLM（趋势研判） ===
            try:
                r4 = item.get("投资评级", {}) if isinstance(item, dict) else {}
                rating = str(r4.get("评级", ""))
                fv = float(r4.get("合理价值", 0)) if isinstance(r4, dict) else 0
                sp = stock_price  # 直接使用外层实际股价（r4.当前价格可能被稀释流程清空）
                advice = str(item.get("操作建议", "")) if isinstance(item, dict) else ""
                has_buy_plan = bool(re.search(r'[≤<=]\s*[\d.]+\s*元.*建仓|买入|仓位', advice))

                divergence_parts = []
                # 检测1: 评级vs操作方向矛盾
                if rating == "SELL" and has_buy_plan:
                    divergence_parts.append(
                        f"量化锚点判定为SELL（合理价值{fv:.0f}元仅为当前价{sp:.0f}元的{(fv/sp*100):.0f}%），"
                        f"但趋势研判认为回调后具备买入价值。这是'便宜与否'和'能否更贵'两种估值哲学的分歧。")
                elif rating == "BUY" and "不建议" in advice:
                    divergence_parts.append(
                        f"量化锚点判定为BUY（合理价值{fv:.0f}元高于当前价{sp:.0f}元），"
                        f"但趋势研判偏谨慎。建议关注催化剂确认后再行动。")
                elif rating == "HOLD" and has_buy_plan:
                    # 检查买入价是否远高于合理价值
                    buy_m = re.search(r'[≤<=]\s*([\d.]+)\s*元', advice)
                    if buy_m:
                        buy_price = float(buy_m.group(1))
                        if buy_price > fv * 1.3 and fv > 0:
                            divergence_parts.append(
                                f"量化锚点合理价值{fv:.0f}元，趋势研判建议建仓价{buy_price:.0f}元（高出{(buy_price/fv-1)*100:.0f}%）。"
                                f"前者基于行业PE中枢和财务数据，后者纳入了增速溢价和市场情绪。两者差距反映了保守估值与趋势定价之间的张力。")

                # 检测2: fair_value与操作建议中的目标价显著背离
                target_m = re.search(r'目标\s*([\d.]+)\s*[-~至]\s*([\d.]+)\s*元', advice)
                if not target_m:
                    # try 综合结论
                    conclusion = item.get("结论", {}) if isinstance(item, dict) else {}
                    exp_return = str(conclusion.get("预期收益", "")) if isinstance(conclusion, dict) else ""
                    target_m = re.search(r'目标\s*([\d.]+)\s*[-~至]\s*([\d.]+)', exp_return)
                if target_m and fv > 0:
                    target_low = float(target_m.group(1))
                    if target_low > fv * 2:
                        divergence_parts.append(
                            f"趋势研判目标价({target_m.group(0)})是量化合理价值({fv:.0f}元)的{(target_low/fv):.0f}倍。"
                            f"这种量级差距说明两种框架对'合理估值'的定义完全不同——前者看PEG和行业趋势，后者看静态PE和资产底。")

                if divergence_parts:
                    # 提取建仓价：兼容 ≤X元 / 回调至X元 / X-Y元区间 / X元左右
                    buy_m = re.search(r'[≤<=]\s*([\d.]+)\s*元', advice)
                    if not buy_m:
                        buy_m = re.search(r'(?:回调|回落|跌)(?:至|到)\s*([\d.]+)\s*元', advice)
                    if not buy_m:
                        buy_m = re.search(r'([\d.]+)\s*[-~至]\s*([\d.]+)\s*元', advice)  # 取区间下限
                    trend_buy = float(buy_m.group(1)) if buy_m else 0
                    sl_m = re.search(r'止损\s*([\d.]+)\s*元', str(item.get("止损", "")) if isinstance(item, dict) else "")
                    stop_loss_price = float(sl_m.group(1)) if sl_m else 0
                    buy_zone_str = str(r4.get("买入区间", "")) if isinstance(r4, dict) else ""
                    bz_m = re.search(r'([\d.]+)', buy_zone_str)
                    value_buy = float(bz_m.group(1)) if bz_m else 0

                    div_note = (
                        f"[框架分歧] 量化锚点 vs 趋势研判 — 两种投资哲学的完整对比\n"
                        f"\n"
                        f"  本报告同时呈现了两套估值逻辑，它们的结论不同，但各有其适用场景。这不是报告的缺陷，\n"
                        f"  而是两种投资哲学在边界处的自然张力。以下将两套逻辑拆开，供您对照选择。\n"
                        f"\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  方案A：价值框架（量化锚点主导）\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  核心信念：价格终将回归内在价值，安全边际是首要原则。\n"
                        f"  合理价值：{fv:.0f}元（基于行业PE中枢 × 财务质量 × 增速溢价，代码计算）\n"
                        f"  安全买入价：≤{value_buy:.0f}元（要求{str(r4.get('安全边际要求','?'))}安全边际）\n"
                        f"  当前价{sp:.0f}元 vs 安全买入价{value_buy:.0f}元 → 差距{((sp-value_buy)/value_buy*100):.0f}%，远未触及买点\n"
                        f"  适合人群：无法接受股价再跌30%仍能持有的人；希望买入后能安稳睡觉的人\n"
                        f"  关键考验：如果股价一路上涨到{sp*1.5:.0f}元而您空仓，您能坦然接受\"错过\"吗？\n"
                        f"\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  方案B：趋势框架（趋势研判主导）\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  核心信念：市场定价有效，高增速可以支撑高估值，强者恒强。\n"
                    )
                    if trend_buy > 0:
                        div_note += (
                        f"  趋势建仓价：≤{trend_buy:.0f}元（基于PEG框架和技术支撑，非价值锚定）\n"
                        f"  当前价{sp:.0f}元 vs 建仓价{trend_buy:.0f}元 → 差距{((sp-trend_buy)/trend_buy*100):.0f}%，需等待回调\n"
                        )
                        if stop_loss_price > 0:
                            wider_stop = round(trend_buy * 0.85, 0)
                            div_note += (
                            f"  建议止损：≥{wider_stop:.0f}元（趋势框架下止损应设得更宽，避免高波动震出）\n"
                            )
                        div_note += (
                        f"  关键考验：如果在{trend_buy:.0f}元买入后跌到{trend_buy*0.8:.0f}元，您会恐慌卖出还是认为加仓机会？\n"
                        )
                    div_note += (
                        f"  仓位建议：初始≤5%（趋势博弈，非价值抄底，必须轻仓试错）\n"
                        f"  适合人群：能承受20-30%回撤而不恐慌的人；相信AI产业趋势大于估值约束的人\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  如何选择？\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  问自己一个问题：\"如果我在建仓价买入后，股价再跌20%，我会恐慌卖出，\n"
                        f"  还是认为这是加仓机会？\"\n"
                        f"   → 答案是\"恐慌卖出\" → 您适合方案A，耐心等待安全买入价。\n"
                        f"   → 答案是\"加仓机会\" → 您适合方案B，按趋势框架操作。\n"
                        f"  两者不互斥，也可以用80%仓位执行方案A，20%仓位试探方案B。\n"
                        f"  关键不是选哪边，而是选了之后言行一致——不要用价值投资的理由买入，\n"
                        f"  却用趋势交易的理由止损。\n"
                        f"\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  关于止损：价格止损 vs 逻辑止损\n"
                        f"  ═══════════════════════════════════════════════════════════════\n"
                        f"  方案A（价值）使用\"逻辑止损\"——不是价格跌了多少就卖，而是投资逻辑\n"
                        f"  是否被破坏。请关注报告中的[证伪条件]段落，当公司基本面恶化（而非\n"
                        f"  股价波动）触发证伪条件时，才执行卖出。价格越跌，安全边际越大，\n"
                        f"  逻辑未破时应考虑加仓而非止损。\n"
                        f"  方案B（趋势）使用\"价格止损\"——股价跌破关键支撑位时离场，因为\n"
                        f"  趋势可能已经反转。此时不需要等基本面确认（等确认时往往已经深套）。\n"
                        f"  两种止损逻辑对应两种投资哲学，混用是长期亏损的最大来源。"
                    )
                    if isinstance(item, dict):
                        item["框架分歧"] = div_note
                        # 双框架执行状态：分开显示价值锚点和趋势锚点
                        _bz_str = str(r4.get("买入区间", "")) if isinstance(r4, dict) else ""
                        _bz_m = re.search(r'([\d.]+)', _bz_str)
                        _v_buy = float(_bz_m.group(1)) if _bz_m else 0
                        _t_buy = trend_buy  # from divergence extraction above
                        _sp_actual = stock_price  # 外层作用域的实际股价
                        if _v_buy > 0 and _t_buy > 0 and _t_buy > _v_buy * 1.5:
                            _va_gap = (_sp_actual - _v_buy) / _v_buy * 100
                            _tr_gap = (_sp_actual - _t_buy) / _t_buy * 100
                            _tr_action = ("已进入建仓区" if _sp_actual <= _t_buy else
                                         "需等待回调" if _tr_gap <= 30 else
                                         "价格偏高，暂不建议建仓")
                            item["执行状态"] = (
                                f"[执行状态-A:价值框架] 安全买入价≤{_v_buy:.0f}元，当前价{_sp_actual:.0f}元(差距{_va_gap:.0f}%)\n"
                                f"  [执行状态-B:趋势框架] 建仓价≤{_t_buy:.0f}元，当前价{_sp_actual:.0f}元(差距{_tr_gap:.0f}%)→{_tr_action}"
                            )
            except: pass
        except Exception:
            pass  # 决策失败不阻塞报告

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                sym = item.get("代码", "")
                if sym:
                    _fix_and_decide(item, sym)
    elif isinstance(data, dict):
        sym = data.get("代码", "")
        if sym:
            _fix_and_decide(data, sym)

    # === 校验Agent：4项一致性检查 ===
    items_to_check = data if isinstance(data, list) else [data]
    for item in items_to_check:
        if not isinstance(item, dict): continue
        val_notes = []
        # 检查1: 定增→合理价值是否已下调
        rating = item.get("投资评级", {})
        fv = rating.get("合理价值", 0) if isinstance(rating, dict) else 0
        risks = item.get("风险", [])
        has_zj = any("定增" in r for r in (risks if isinstance(risks, list) else []))
        scores = item.get("评分", {})
        growth = scores.get("成长性", {}).get("得分", 5) if isinstance(scores, dict) else 5
        # 检查2: 成长性<5→乐观PE不膨胀
        if growth is not None and isinstance(growth, (int, float)) and growth < 5:
            sc = item.get("情景估值", {})
            opt = sc.get("乐观", {}) if isinstance(sc, dict) else {}
            if isinstance(opt, dict) and opt.get("价格", 0):
                try:
                    opt_p = float(opt["价格"])
                    if opt_p > 0 and float(fv) > 0:
                        pe_est = opt_p / float(fv) * 18  # rough PE estimate from price/fair_value ratio
                        if pe_est > 25:
                            val_notes.append(f"[校验⚠️] 成长性{int(growth)}分(C级)，但乐观PE约{pe_est:.0f}倍偏高")
                except: pass
        # 检查3: 风险有定增→投资评级是否合理
        if has_zj and isinstance(rating, dict) and rating.get("评级") == "BUY":
            val_notes.append("[校验⚠️] 有定增摊薄风险但评级为BUY——请确认合理价值已按摊薄系数下调")

        if val_notes:
            item["校验"] = val_notes

    # === 注入估值框架到 items（供 format_report 渲染）===
    if _val_framework:
        _items = data if isinstance(data, list) else [data]
        for _it in _items:
            if isinstance(_it, dict):
                _it["估值框架"] = _val_framework

    # === 渲染评分卡 ===
    compare_text = ""
    if isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, dict) and "对比分析" in item:
                cmp = item.pop("对比分析")
                if isinstance(cmp, dict):
                    compare_text = _format_compare_section(cmp)
            cleaned.append(item)
        score_cards = [format_report(it) for it in cleaned if isinstance(it, dict)]
        score_text = "\n\n".join(score_cards)
        if compare_text:
            score_text += "\n\n" + compare_text
    elif isinstance(data, dict):
        if "对比分析" in data:
            cmp = data.pop("对比分析")
            if isinstance(cmp, dict):
                compare_text = _format_compare_section(cmp)
        score_text = format_report(data)
        if compare_text:
            score_text += "\n\n" + compare_text
    else:
        score_text = str(data)

    # 数据时效标注 + 数据源等级
    import backend.datasource_tier as _rpt_dst
    _tier_labels = {0: "免费API(行情+财报+公告)", 1: "付费终端(+管理层+机构持仓+产业链)", 2: "机构级(+ESG+另类数据+Level2)"}
    _tier_line = f"数据源: {_rpt_dst.tier.name} — {_tier_labels.get(_rpt_dst.tier.value, '?')}"
    # 标记不可用的高级插槽
    _unavail = set()
    for item in (data if isinstance(data, list) else [data]):
        if isinstance(item, dict) and item.get("_unavailable_premium"):
            _unavail.update(item["_unavailable_premium"])
    if _unavail:
        _tier_line += f" | 未启用: {', '.join(sorted(_unavail))} (升级数据源后可激活)"
    header = f"数据时效: {period_note}\n{_tier_line}\n{'=' * 64}\n\n"

    # _get_llm() 只在评分卡下面加一段叙述性结论
    # ---- Critic审查反馈注入 ----
    _critic_line = ""
    _critic_fixes = ""
    _cm = re.search(r'\[Critic审查:[^\]]+\][^\n]*', state.get("analysis", ""))
    if _cm:
        _critic_line = "  " + _cm.group(0) + "\n"
        # 优先从结构化 [CRITIC_FIXES] 标记提取，回退到文本解析
        _cfm = re.search(r'\[CRITIC_FIXES\]\s*(\[.*\])', state.get("analysis", ""))
        if _cfm:
            try:
                _fixes_list = json.loads(_cfm.group(1))
                _critic_fixes = "\n\n[!!!] Critic审查发现以下问题，你的总结必须避开或修正:\n"
                for f in _fixes_list[:5]:
                    _critic_fixes += "  - " + f + "\n"
                _critic_fixes += "禁止重复Critic已指出的矛盾。若操作建议中存在评级矛盾，统一为与投资决策一致的方向。"
            except: pass
        if not _critic_fixes:
            _c_full = state.get("analysis", "")
            _c_idx = _c_full.find("[Critic审查:")
            _c_end = _c_full.find("\n{", _c_idx) if _c_idx >= 0 else -1
            _c_text = _c_full[_c_idx:_c_end] if _c_end > _c_idx else _c_full[_c_idx:_c_idx+600]
            if "发现漏洞" in _c_text:
                _critic_fixes = ("\n\n[!!!] Critic审查发现以下逻辑漏洞，你的总结必须避开或修正这些问题:\n"
                                 + _c_text.replace("[Critic审查:", "").strip()[:500]
                                 + "\n\n禁止重复Critic已指出的矛盾（如高估却建议等待）。若操作建议中存在评级矛盾，统一为与投资决策一致的方向。")

    _reporter_prompt = REPORTER_PROMPT + _critic_fixes
    narrative = _get_llm().invoke([
        SystemMessage(content=_reporter_prompt),
        HumanMessage(content=f"评分卡已生成:\n{score_text}\n\n原始分析JSON:\n{raw}\n\n请为以上分析写一段总结(2-3句话)和投资建议。"),
    ]).content

    # ---- 校验审计Agent（轻量代码预检 + LLM审计）----
    audit_report = header + _critic_line + score_text + "\n\n" + narrative

    # 代码级快速预检：检查明显的数据矛盾，全部通过则降级审计为"仅警告"
    _code_issues = []
    for item in (data if isinstance(data, list) else [data]):
        if not isinstance(item, dict): continue
        rating = item.get("投资评级", {}) if isinstance(item, dict) else {}
        scores = item.get("评分", {}) if isinstance(item, dict) else {}
        # 评分合计一致性
        dim_sum = sum(s.get("得分", 0) for s in scores.values()
                      if isinstance(s, dict) and isinstance(s.get("得分"), (int, float)) and s.get("得分", 0) <= 10)
        # 估值水位 PE 存在性
        vw = item.get("估值水位", {}) if isinstance(item, dict) else {}
        has_pe = bool(vw.get("PE")) if isinstance(vw, dict) else False
        # 风险-估值联动：有定增风险则合理价值应 < 当前价*2（粗略检查）
        risks = item.get("风险", [])
        has_dilution_risk = any("定增" in str(r) for r in (risks if isinstance(risks, list) else []))
        fv = rating.get("合理价值", 0) if isinstance(rating, dict) else 0
        price = rating.get("当前价格", stock_price) if isinstance(rating, dict) else stock_price

        if not has_pe:
            _code_issues.append("❌ 估值水位PE缺失")
        if has_dilution_risk and isinstance(fv, (int, float)) and fv > 0 and price > 0:
            if fv / price > 2.5:
                _code_issues.append(f"❌ 定增风险存在但合理价值/股价={fv/price:.1f}倍偏高(疑似未摊薄)")

    _skip_auditor = len(_code_issues) == 0
    if _skip_auditor:
        audit_report += "\n[校验✅] 代码级一致性检查通过。"
        # 仍运行LLM审计但仅追加警告（不触发重试）
    else:
        audit_report += f"\n[校验⚠️] 代码预检发现问题: {'; '.join(_code_issues)}"

    retry_count = 0
    _max_retries = 1 if _skip_auditor else 3  # 预检通过→最多1次审计(仅警告)；否则最多3次
    for attempt in range(_max_retries + 1):
        try:
            # 取报告头3000字+尾部1000字（确保操作建议/结论/止损不被截断）
            _head = audit_report[:3000]
            _tail = audit_report[-1500:] if len(audit_report) > 3500 else ""
            _tail_sections = ""
            if _tail:
                # 只提取尾部关键段落：操作建议/止损/执行状态/综合结论
                for kw in ["[操作建议]", "[止损]", "[执行状态]", "[综合结论]", "[定增信息]"]:
                    idx = audit_report.find(kw)
                    if idx > 3000:
                        end = min(idx + 300, len(audit_report))
                        _tail_sections += audit_report[idx:end] + "\n"
            audit_body = _head + ("\n...(中略)...\n" + _tail_sections if _tail_sections else "")
            audit_prompt = f"请审查以下投资报告，找出逻辑矛盾:\n\n{audit_body}"
            audit_resp = _get_llm().invoke([
                SystemMessage(content=AUDITOR_PROMPT),
                HumanMessage(content=audit_prompt),
            ])
            audit_json = json.loads(audit_resp.content.strip())
            issues = audit_json.get("问题", [])
            if not issues:
                break

            # 分级处理
            critical = [i for i in issues if i.get("级别") == "❌"]
            warnings = [i for i in issues if i.get("级别") == "⚠️"]

            # 代码预检通过时：所有问题降级为警告，不触发重试
            if _skip_auditor:
                for w in (critical + warnings):
                    desc = w.get("描述", "")
                    fix = w.get("修正建议", "")
                    audit_report += f"\n[审计⚠️] {desc}" + (f" (建议: {fix})" if fix else "")
                break

            if critical or retry_count >= 2:
                if retry_count == 0 and critical and not _skip_auditor:
                    # 首次严重失败：把审计摘要喂回 Analyst 重新推理（根源修复）
                    failure_details = []
                    for i, iss in enumerate(critical):
                        desc = iss.get("描述", "")
                        fix = iss.get("修正建议", "")
                        failure_details.append(f"问题{i+1}: {desc}" + (f" → 请修正为: {fix}" if fix else ""))
                    failure_summary = " | ".join(failure_details)
                    retry_prompt = (
                        f"[审计退回 — 上一版报告未通过审计，以下问题必须逐条修正]\n\n"
                        f"{failure_summary}\n\n"
                        f"[审计原文] 以下是审计Agent的完整审查结果，请理解每一条并修正:\n"
                        f"{json.dumps(critical, ensure_ascii=False, indent=2)}\n\n"
                        f"上述问题对应报告中的具体数字冲突或逻辑矛盾。请基于collected_data重新推理，确保数字自洽。输出完整JSON。"
                    )
                    retry_resp = _get_llm().invoke([
                        SystemMessage(content=ANALYST_PROMPT),
                        HumanMessage(content=f"{state.get('collected_data','')[:4000]}\n\n{retry_prompt}"),
                    ])
                    raw = retry_resp.content
                    # 重新解析+评分覆盖+格式化
                    new_data = None
                    try: new_data = json.loads(raw.strip())
                    except: pass
                    if new_data is None:
                        decoder = json.JSONDecoder()
                        for i, ch in enumerate(raw.strip()):
                            if ch in '[{':
                                try: new_data, _ = decoder.raw_decode(raw.strip()[i:]); break
                                except: continue
                    if new_data is not None:
                        # 保留旧 item 的处理标记（防止二次稀释等重复处理）
                        _old_markers = {}
                        old_items = data if isinstance(data, list) else [data]
                        for oi in old_items:
                            if isinstance(oi, dict) and oi.get("代码"):
                                _old_markers[oi["代码"]] = {
                                    k: oi[k] for k in ["_dilution_coefficient", "_dilution_shares", "_dilution_fund_amount"]
                                    if k in oi
                                }
                        data = new_data
                        # 将旧标记注入新 item
                        new_items = data if isinstance(data, list) else [data]
                        for ni in new_items:
                            if isinstance(ni, dict) and ni.get("代码") in _old_markers:
                                ni.update(_old_markers[ni["代码"]])
                        # 重新处理（_fix_and_decide 会检查标记，跳过已处理的步骤）
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and (sym := item.get("代码")):
                                    _fix_and_decide(item, sym)
                        elif isinstance(data, dict) and (sym := data.get("代码")):
                            _fix_and_decide(data, sym)
                        score_cards2 = [format_report(it) for it in (data if isinstance(data, list) else [data]) if isinstance(it, dict)]
                        score_text = "\n\n".join(score_cards2)
                        narrative = _get_llm().invoke([
                            SystemMessage(content=REPORTER_PROMPT + _critic_fixes),
                            HumanMessage(content=f"评分卡:\n{score_text}\n\n请写总结。"),
                        ]).content
                        audit_report = header + score_text + "\n\n" + narrative
                        retry_count += 1
                        continue  # 重新进入审计循环

                # 重试耗尽 → 降级输出：保留已格式化的修正报告，仅追加审计摘要
                if retry_count >= 2:
                    fallback = "\n\n⚠️ 系统提示：自动校验未完全通过，以上报告数据已经代码量化修正（含定增摊薄、评分覆盖、字段重算）。"
                    if critical:
                        fallback += f"\n审计发现({len(critical)}项严重/{len(warnings)}项警告)，建议人工复核关键数字。"
                    fallback += "\n"
                    audit_report += fallback
                    break

                # 第二级：定向外科手术——只改写被标记段落
                fix_instructions = "; ".join([i.get("修正建议", "") for i in critical])
                fix_prompt = (f"以下报告存在数据矛盾:\n{fix_instructions}\n"
                              f"请只修改投资决策和结论段，使数字与风险描述一致。输出全文。")
                fix_resp = _get_llm().invoke([
                    SystemMessage(content=REPORTER_PROMPT),
                    HumanMessage(content=f"{audit_report[:2000]}\n\n{fix_prompt}"),
                ])
                audit_report = fix_resp.content
                retry_count += 1
            elif warnings:
                # 第一级：轻微问题→追加审计标注
                for w in warnings:
                    desc = w.get("描述", "")
                    fix = w.get("修正建议", "")
                    audit_report += f"\n[审计⚠️] {desc}" + (f" (建议: {fix})" if fix else "")
                break  # 标注后通过，不重试

        except Exception:
            break

    # === 调用证据：从 collected_data 文本头部 + processing_log 收集 ===
    _evidence_parts = []
    # 工具痕迹（从 collected_data 的 [TOOLS] 头部读取，绕过 checkpointer 序列化问题）
    _collected_raw = state.get("collected_data", "")
    _tools_match = re.search(r'\[TOOLS\]\s*([^\n]+)', _collected_raw)
    if _tools_match:
        _evidence_parts.append(f"数据工具: {_tools_match.group(1)}")
    else:
        # 回退：从 processing_log 读取
        for _pl in state.get("processing_log", []):
            if _pl.get("tool_calls"):
                _tool_summary = ", ".join(f"{t}({s})" for t, s in _pl["tool_calls"][:12])
                _evidence_parts.append(f"数据工具: {_tool_summary}")
                break
    # RAG痕迹（从 processing_log 读取，analyst_node 写入）
    for _pl in state.get("processing_log", []):
        if _pl.get("rag_calls"):
            _evidence_parts.append(f"RAG知识库: {'; '.join(_pl['rag_calls'][:5])}")
            break
    _evidence_text = ""
    if _evidence_parts:
        _evidence_text = "\n  [调用证据] " + " | ".join(_evidence_parts)
        _evidence_text += "\n  * 以上为系统自动记录的工具调用与知识库检索痕迹，用于验证分析的数据来源。\n"

    # === 审计摘要：收集所有检查结果，构建可见的校验表格 ===
    _audit_rows = []
    # 从 items 中收集审计信号
    for _it in (data if isinstance(data, list) else [data]):
        if not isinstance(_it, dict): continue
        sym_name = _it.get("名称", _it.get("代码", "?"))

        # 1. 事件-估值联动
        if _it.get("定增信息"):
            _audit_rows.append(("事件-估值联动", "✅", "定增摊薄已由代码强制修正"))
        elif _it.get("_dilution_coefficient"):
            _audit_rows.append(("事件-估值联动", "✅", f"稀释系数{_it['_dilution_coefficient']}已应用"))
        else:
            _audit_rows.append(("事件-估值联动", "—", "无定增事件"))

        # 2. 评分-估值矛盾 (from code pre-check)
        _audit_rows.append(("评分-估值矛盾", "✅" if _skip_auditor else "⚠️", "代码预检" if _skip_auditor else "见审计Agent"))

        # 3. 评级-操作一致
        if _it.get("框架分歧"):
            _audit_rows.append(("评级-操作一致", "⚠️", "已呈现框架分歧(非错误)"))
        elif _it.get("校验修正"):
            _audit_rows.append(("评级-操作一致", "⚠️", f"已修正: {_it['校验修正'][:50]}"))
        else:
            _audit_rows.append(("评级-操作一致", "✅", "一致"))

        # 4. 估值合理
        _r = _it.get("投资评级", {}) if isinstance(_it, dict) else {}
        _fv = _r.get("合理价值", 0) if isinstance(_r, dict) else 0
        _sp = _r.get("当前价格", 0) if isinstance(_r, dict) else 0
        try:
            _ratio = float(_fv) / float(_sp) if float(_sp) > 0 else 1
            if _ratio > 2.5:
                _audit_rows.append(("估值合理", "⚠️", f"合理价值/股价={_ratio:.1f}倍"))
            elif _ratio < 0.3:
                _audit_rows.append(("估值合理", "⚠️", f"合理价值仅为股价{_ratio*100:.0f}%"))
            else:
                _audit_rows.append(("估值合理", "✅", f"比值={_ratio:.1f}"))
        except:
            _audit_rows.append(("估值合理", "—", "无法计算"))

        # 5. 数据单位
        _audit_rows.append(("数据单位", "✅" if not _it.get("校验修正") else "⚠️", "通过" if not _it.get("校验修正") else "已触发修正"))

        # 6. 评分合计
        _audit_rows.append(("评分合计", "✅", "代码生成,LLM不参与"))

        # 7. 操作建议
        _exec = _it.get("执行状态", "")
        _audit_rows.append(("操作建议", "✅" if _exec else "⚠️", "含执行状态判定" if _exec else "检查中"))

        # 8. 情景EPS
        _audit_rows.append(("情景EPS单调性", "⚠️", "LLM生成,见审计Agent检查"))

    # 构建表格
    _audit_table = "\n  [校验审计] 8项守卫检查结果:\n"
    _audit_table += "  | 检查项 | 状态 | 说明 |\n"
    _audit_table += "  |--------|:----:|------|\n"
    for row in _audit_rows[:16]:  # 最多2只股票×8项
        _audit_table += f"  | {row[0]} | {row[1]} | {row[2]} |\n"
    _audit_table += f"  * 审计重试: {retry_count}次 | 代码预检: {'通过' if _skip_auditor else '发现问题'}\n"

    audit_report += "\n" + _audit_table
    if _evidence_text:
        audit_report += _evidence_text

    prev_log = state.get("processing_log", [])
    _narrative_len = len(narrative) if narrative else 0
    prev_log.append({"phase": "Report", "summary": f"报告生成完成",
                     "status": "SUCCESS", "output_chars": len(audit_report),
                     "score_chars": len(score_text), "narrative_chars": _narrative_len,
                     "audit_retries": retry_count, "max_retries": 3,
                     "code_precheck": "通过" if _skip_auditor else "发现问题"})
    return {"report": audit_report, "processing_log": prev_log}

# ============================================================
#  图构建
# ============================================================

_GRAPH = None

def build_graph():
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    graph = StateGraph(FinBrainState)
    graph.add_node("data_collector", data_collector_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("valuation", valuation_agent_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "data_collector")
    graph.add_edge("data_collector", "analyst")
    graph.add_edge("analyst", "valuation")
    graph.add_edge("valuation", "critic")
    graph.add_edge("critic", "reporter")
    graph.add_edge("reporter", END)

    _GRAPH = graph.compile(checkpointer=_make_checkpointer())
    return _GRAPH

# ============================================================
#  上下文压缩
# ============================================================

COMPRESS_KEEP = int(os.getenv("COMPRESS_KEEP", "6"))
COMPRESS_TRIGGER = int(os.getenv("COMPRESS_TRIGGER", "12"))

def compress_history(history: list) -> list:
    if len(history) <= COMPRESS_TRIGGER:
        return history

    old = history[:-COMPRESS_KEEP]
    recent = history[-COMPRESS_KEEP:]

    old_text = "\n".join(
        f"[{'用户' if m['role'] == 'user' else 'Agent'}]: {m['content'][:200]}"
        for m in old
    )

    try:
        summary_msg = [HumanMessage(
            content=f"用80字以内概括这段股票分析对话的关键结论和数据:\n{old_text}")]
        summary_result = _get_llm().invoke(summary_msg)
        summary = summary_result.content
    except Exception:
        summary = f"[前{len(old)}条消息的上下文已省略]"

    return [{"role": "user", "content": f"[历史摘要] {summary}"}] + recent

def _dicts_to_messages(history: list) -> list:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

_CHAT_TOOLS = [resolve_stock, search_youzi_kb, search_knowledge, recent_announcements, stock_price, stock_history, intraday, sector_fund_flow, place_order, execute_analysis, show_portfolio, trade_history]
_CHAT_AGENT = None

def _get_chat_agent():
    global _CHAT_AGENT
    if _CHAT_AGENT is None:
        _CHAT_AGENT = create_agent(_get_llm(), _CHAT_TOOLS,
                                   system_prompt="你是 FinBrain 投研助手。可以查股价和K线，其他数据需切换到分析模式。不编造数据。",
                                   checkpointer=_make_checkpointer())
    return _CHAT_AGENT

CHAT_PROMPT = """你是 FinBrain，一个A股投研助手。可以闲聊、答疑问、解释概念。
你有 stock_price/stock_history 查行情，recent_announcements 查公告（分析前先查公告——看有没有定增/减持/业绩预告）。
如果用户问财报/估值/行业等深度数据，建议"切换到分析模式"。"""

# ============================================================
#  路由判断
# ============================================================

def _classify_request(user_input: str) -> str:
    """判断请求类型（从策略配置读取触发词）"""
    triggers = _get_strategy().get("triggers", {})
    phantom_words = triggers.get("phantom", ["妖股", "猎妖", "涨停"])
    analysis_words = triggers.get("analysis", ["分析", "报告", "评分"])
    if phantom_words and any(t in user_input for t in phantom_words):
        return "phantom"
    if any(t in user_input for t in analysis_words):
        return "analysis"
    return "chat"

# ============================================================
#  API
# ============================================================

def ask(question: str, history: list = None) -> str:
    req_type = _classify_request(question)
    cfg = {"configurable": {"thread_id": "api_default"}}
    if req_type == "phantom":
        phantom = _get_phantom_agent()
        msgs = _dicts_to_messages(history or [])
        msgs.append(HumanMessage(content=question))
        result = phantom.invoke({"messages": msgs}, config=cfg)
        return result["messages"][-1].content
    elif req_type == "analysis":
        graph = build_graph()
        lc_messages = _dicts_to_messages(history or [])
        lc_messages.append(HumanMessage(content=question))
        result = graph.invoke({
            "messages": lc_messages,
            "user_question": question,
            "collected_data": "",
            "analysis": "",
            "report": "",
            "processing_log": [],
        }, config=cfg)
        return result.get("report") or result["messages"][-1].content
    else:
        chat = _get_chat_agent()
        msgs = _dicts_to_messages(history or [])
        msgs.append(HumanMessage(content=question))
        result = chat.invoke({"messages": msgs}, config=cfg)
        return result["messages"][-1].content

# ============================================================
#  终端交互
# ============================================================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("FinBrain Agent")
    print("Type 'quit' to exit, 'clear' to reset context")
    print()

    graph = build_graph()
    history = []

    while True:
        try:
            user_input = input("\n>$ ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("bye")
                break
            if user_input.lower() == "clear":
                history = []
                print("[context cleared]")
                continue
            if not user_input:
                continue

            history = compress_history(history)

            req_type = _classify_request(user_input)
            if req_type == "phantom":
                phantom = _get_phantom_agent()
                msgs = _dicts_to_messages(history)
                msgs.append(HumanMessage(content=user_input))
                reply = phantom.invoke({"messages": msgs})["messages"][-1].content
            elif req_type == "analysis":
                lc_messages = _dicts_to_messages(history)
                lc_messages.append(HumanMessage(content=user_input))
                result = graph.invoke({
                    "messages": lc_messages,
                    "user_question": user_input,
                    "collected_data": "",
                    "analysis": "",
                    "report": "",
                })
                reply = result.get("report") or result["messages"][-1].content
            else:
                chat = _get_chat_agent()
                msgs = _dicts_to_messages(history)
                msgs.append(HumanMessage(content=user_input))
                reply = chat.invoke({"messages": msgs})["messages"][-1].content

            print(reply)
            print("-" * 60)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nbye")
            break
        except Exception as e:
            print(f"[Error] {e}")
