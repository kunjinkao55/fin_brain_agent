"""
FinBrain — 多 Agent 协作财报分析系统
Data Collector → Analyst → Reporter (LangGraph StateGraph)
"""

import json, os, re
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 从项目根目录加载.env（兼容从任意目录运行）
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
load_dotenv(_env_path)

# 加载策略配置
def _load_strategy():
    strat_file = os.path.join(os.path.dirname(__file__), "..", "configs", "strategies.json")
    with open(strat_file, "r", encoding="utf-8") as f:  # already correct
        all_strategies = json.load(f)
    active = os.getenv("FINBRAIN_STRATEGY", "default")
    if active not in all_strategies:
        active = "default"
    return all_strategies[active], all_strategies

_STRATEGY, ALL_STRATEGIES = _load_strategy()

# ============================================================
#  LLM 初始化（多提供商切换）
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

if LLM_PROVIDER == "deepseek":
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(
        model="deepseek-chat", temperature=0, max_tokens=4096,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
elif LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(
        model="gpt-4o", temperature=0, max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:
    from langchain_anthropic import ChatAnthropic
    LLM = ChatAnthropic(
        model="claude-sonnet-5", temperature=0, max_tokens=4096,
    )

# ============================================================
#  State Schema
# ============================================================

class FinBrainState(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str
    collected_data: str
    analysis: str
    report: str

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
def limit_up_pool(top_n: int = 30) -> str:
    """获取今日涨停板股票池。返回涨停股列表及涨幅/换手率/市值。参数: top_n=返回数量"""
    from backend.tools import get_limit_up_pool
    result = get_limit_up_pool(top_n)
    if "列表" in result and len(result["列表"]) > 15:
        # 只返回前15只给LLM，节省token
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
def resolve_stock(query: str) -> str:
    """根据股票名称或代码模糊搜索，返回匹配的代码-名称列表。如 '新易盛' → 300502"""
    from backend.stock_map import fuzzy_search
    results = fuzzy_search(query, limit=5)
    if not results:
        return json.dumps({"error": f"未找到匹配 '{query}' 的股票"}, ensure_ascii=False)
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

# ---- 妖股猎人工具集 ----
_PHANTOM_TOOLS = [resolve_stock, limit_up_pool, concept_ranking, dragon_tiger_list, dragon_tiger_detail,
                  stock_price, stock_history, fund_flow, financial_statements, valuation, industry_info,
                  place_order, execute_analysis, show_portfolio, trade_history]
_PHANTOM_AGENT = None

def _get_phantom_agent():
    global _PHANTOM_AGENT
    if _PHANTOM_AGENT is None:
        _PHANTOM_AGENT = create_agent(LLM, _PHANTOM_TOOLS, system_prompt=_STRATEGY.get("phantom", ""))
    return _PHANTOM_AGENT

# ============================================================
#  Prompt 分解
# ============================================================

DATA_COLLECTOR_PROMPT = _STRATEGY["data_collector"]

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
  "操作建议": "一句话",
  "止损": "条件",
  "结论": {
    "总评": "财报质量+估值+长线判断",
    "买入策略": "具体价位和仓位",
    "持有策略": "一句话",
    "卖出条件": "止损条件",
    "预期收益": "目标价位",
    "持仓周期": "预计时长"
  },
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

ANALYST_PROMPT = _STRATEGY["analyst"] + _FORMAT_MANDATORY

# ============================================================
#  节点函数
# ============================================================

_data_collector_tools = [resolve_stock, stock_price, stock_history, financial_statements,
                         valuation, industry_info, screen_stocks, fund_flow,
                         place_order, execute_analysis, show_portfolio, trade_history]

_COLLECTOR_AGENT = None

def _get_collector():
    global _COLLECTOR_AGENT
    if _COLLECTOR_AGENT is None:
        _COLLECTOR_AGENT = create_agent(LLM, _data_collector_tools,
                                        system_prompt=DATA_COLLECTOR_PROMPT)
    return _COLLECTOR_AGENT

def data_collector_node(state: FinBrainState) -> dict:
    """调用工具搜集数据（带对话历史上下文）"""
    collector = _get_collector()
    # 把历史消息 + 当前问题一起传给子Agent，让它知道上下文
    msgs = list(state.get("messages", []))
    msgs.append(HumanMessage(content=state["user_question"]))
    result = collector.invoke({"messages": msgs})
    collected = result["messages"][-1].content
    return {"collected_data": collected}

def analyst_node(state: FinBrainState) -> dict:
    """分析数据，输出结构化JSON"""
    prompt = (
        f"用户问题: {state['user_question']}\n\n"
        f"=== 已搜集数据 ===\n{state['collected_data']}\n\n"
        f"请按评分框架分析，输出JSON。"
    )
    response = LLM.invoke([
        SystemMessage(content=ANALYST_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {"analysis": response.content}

REPORTER_PROMPT = """你是 FinBrain 报告格式化专员。

将分析JSON格式化为可读报告。评分卡和表格会由代码自动生成，你只需要写:
- 单只股票: 结论段(1-2段，含投资建议)
- 多只股票: 排名总评 + 总结建议
不使用emoji，不使用markdown加粗。"""

def reporter_node(state: FinBrainState) -> dict:
    """代码生成评分卡（对齐表格）+ LLM生成叙述"""
    from backend.tools import format_report
    raw = state.get("analysis", "")

    if not raw.strip():
        return {"report": "[无分析数据]"}

    # 解析JSON
    raw_stripped = raw.strip()
    data = None
    for attempt in [raw_stripped,
                    re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_stripped),
                    re.search(r'\[[\s\S]*\]', raw_stripped),
                    re.search(r'\{[\s\S]*\}', raw_stripped)]:
        try:
            if isinstance(attempt, str):
                data = json.loads(attempt)
            elif attempt:
                data = json.loads(attempt.group(1) if attempt.lastindex else attempt.group())
            break
        except (json.JSONDecodeError, AttributeError):
            continue

    if data is None:
        # 解析失败，退回纯LLM
        response = LLM.invoke([
            SystemMessage(content="请将以下分析格式化为可读报告"),
            HumanMessage(content=raw),
        ])
        return {"report": response.content}

    # 单只 or 多只: 代码生成评分卡
    if isinstance(data, list):
        score_cards = [format_report(item) for item in data if isinstance(item, dict)]
        score_text = "\n\n".join(score_cards)
    else:
        score_text = format_report(data)

    # LLM 只在评分卡下面加一段叙述性结论
    narrative = LLM.invoke([
        SystemMessage(content=REPORTER_PROMPT),
        HumanMessage(content=f"评分卡已生成:\n{score_text}\n\n原始分析JSON:\n{raw}\n\n请为以上分析写一段总结(2-3句话)和投资建议。"),
    ]).content

    return {"report": score_text + "\n\n" + narrative}

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
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "data_collector")
    graph.add_edge("data_collector", "analyst")
    graph.add_edge("analyst", "reporter")
    graph.add_edge("reporter", END)

    _GRAPH = graph.compile()
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
        summary_msg = [{"role": "user",
                        "content": f"用80字以内概括这段股票分析对话的关键结论和数据:\n{old_text}"}]
        summary_result = LLM.invoke(summary_msg)
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

_CHAT_TOOLS = [resolve_stock, stock_price, stock_history, place_order, execute_analysis, show_portfolio, trade_history]
_CHAT_AGENT = None

def _get_chat_agent():
    global _CHAT_AGENT
    if _CHAT_AGENT is None:
        _CHAT_AGENT = create_agent(LLM, _CHAT_TOOLS,
                                   system_prompt="你是 FinBrain 投研助手。可以查股价和K线，其他数据需切换到分析模式。不编造数据。")
    return _CHAT_AGENT

CHAT_PROMPT = """你是 FinBrain，一个A股投研助手。可以闲聊、答疑问、解释概念。
你手头有 stock_price 和 stock_history 工具，可以查股价和K线。
如果用户问财报/估值/行业/资金流向等深度数据，建议"切换到分析模式"。"""

# ============================================================
#  路由判断
# ============================================================

def _classify_request(user_input: str) -> str:
    """判断请求类型（从策略配置读取触发词）"""
    triggers = _STRATEGY.get("triggers", {})
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

def _dicts_to_messages(history: list) -> list:
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

def ask(question: str, history: list = None) -> str:
    req_type = _classify_request(question)
    if req_type == "phantom":
        phantom = _get_phantom_agent()
        msgs = _dicts_to_messages(history or [])
        msgs.append(HumanMessage(content=question))
        result = phantom.invoke({"messages": msgs})
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
        })
        return result.get("report") or result["messages"][-1].content
    else:
        chat = _get_chat_agent()
        msgs = _dicts_to_messages(history or [])
        msgs.append(HumanMessage(content=question))
        result = chat.invoke({"messages": msgs})
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
