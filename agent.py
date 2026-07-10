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

load_dotenv()

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
    from tools import fetch_stock_price
    return json.dumps(fetch_stock_price(symbol), ensure_ascii=False, indent=2)

@tool
def stock_history(symbol: str, scale: int = 240, datalen: int = 30) -> str:
    """查询A股历史K线数据。symbol:股票代码 scale:240=日线 datalen:条数"""
    from tools import fetch_stock_history
    result = fetch_stock_history(symbol, scale, datalen)
    if "data" in result and len(result["data"]) > 10:
        result["data"] = result["data"][-10:]
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def financial_statements(symbol: str) -> str:
    """查询近三年三大报表：利润表、资产负债表、现金流量表。输入股票代码"""
    from tools import get_financial_statements
    return json.dumps(get_financial_statements(symbol), ensure_ascii=False, indent=2)

@tool
def valuation(symbol: str) -> str:
    """查询估值指标：ROE、毛利率、净利率、每股收益、资产负债率等。输入股票代码"""
    from tools import get_valuation
    return json.dumps(get_valuation(symbol), ensure_ascii=False, indent=2)

@tool
def industry_info(symbol: str) -> str:
    """查询个股所属行业分类及行业指数表现。输入股票代码"""
    from tools import get_industry_info
    return json.dumps(get_industry_info(symbol), ensure_ascii=False, indent=2)

@tool
def fund_flow(symbol: str) -> str:
    """查询个股当日资金流向：主力净流入/流出/净额。输入股票代码如 601991"""
    from tools import get_fund_flow
    return json.dumps(get_fund_flow(symbol), ensure_ascii=False, indent=2)

@tool
def limit_up_pool(top_n: int = 30) -> str:
    """获取今日涨停板股票池。返回涨停股列表及涨幅/换手率/市值。参数: top_n=返回数量"""
    from tools import get_limit_up_pool
    result = get_limit_up_pool(top_n)
    if "列表" in result and len(result["列表"]) > 15:
        # 只返回前15只给LLM，节省token
        result["列表"] = result["列表"][:15]
    return json.dumps(result, ensure_ascii=False, indent=2)

@tool
def concept_ranking(top_n: int = 20) -> str:
    """获取同花顺概念板块列表。返回概念名称和代码。参数: top_n=返回数量"""
    from tools import get_concept_ranking
    return json.dumps(get_concept_ranking(top_n), ensure_ascii=False, indent=2)


@tool
def screen_stocks(max_pe: float = 30, max_pb: float = 5,
                  min_mktcap: float = 20, top_n: int = 30) -> str:
    """全市场扫描，返回PE从低到高排列的A股列表。
    参数: max_pe=市盈率上限, max_pb=市净率上限, min_mktcap=最低市值(亿), top_n=返回条数"""
    from tools import screen_stocks as do_screen
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
    from tools import get_dragon_tiger_list
    return json.dumps(get_dragon_tiger_list(date), ensure_ascii=False, indent=2)

@tool
def dragon_tiger_detail(symbol: str) -> str:
    """查询个股龙虎榜买卖席位明细，识别知名游资（炒股养家/方新侠/上塘路等）。输入股票代码"""
    from tools import get_dragon_tiger_detail
    return json.dumps(get_dragon_tiger_detail(symbol), ensure_ascii=False, indent=2)

# ---- 妖股猎人工具集 ----
_PHANTOM_TOOLS = [limit_up_pool, concept_ranking, dragon_tiger_list, dragon_tiger_detail,
                  stock_price, stock_history, fund_flow, financial_statements, valuation, industry_info]
_PHANTOM_AGENT = None

def _get_phantom_agent():
    global _PHANTOM_AGENT
    if _PHANTOM_AGENT is None:
        _PHANTOM_AGENT = create_agent(LLM, _PHANTOM_TOOLS, system_prompt="""你是 FinBrain 妖股猎人 Phantom Hunter。专门识别短期爆发力极强的妖股。

妖股特征:
- 市值 20-80亿, 股价<20元, 小盘易拉抬
- 涨停板启动: 今日涨停且换手率5-20%(拒绝一字板无量)
- 题材硬: 属于当下最热概念板块
- 龙虎榜游资: 知名游资(炒股养家/方新侠等)买入比卖出多

工作流:
1. 调 limit_up_pool 获取今日涨停池
2. 调 dragon_tiger_list 获取龙虎榜上榜股
3. 涨停股与龙虎榜交叉, 对交集股调 dragon_tiger_detail 看游资
4. 调 fund_flow + concept_ranking 验证资金和题材
5. 输出《妖股猎手内参》:
   - 排名/代码/名称/题材/市值/换手率
   - 游资标签(格局/一日游)
   - 博弈买点(首阴/深水急刹)
   - 止损线(-5%或10日线)
   - 仓位: 单只不超过总资产5%

严禁推荐ST/*ST/北交所/市值超200亿.
""")
    return _PHANTOM_AGENT

# ============================================================
#  Prompt 分解
# ============================================================

DATA_COLLECTOR_PROMPT = """你是 FinBrain 数据搜集专员。你的唯一工作是调用工具获取数据。

单只股票：调满这6个工具，缺一不可:
  1. stock_price(代码)
  2. stock_history(代码)
  3. financial_statements(代码)
  4. valuation(代码)
  5. industry_info(代码)
  6. fund_flow(代码)

多只或扫描：如果调用了 screen_stocks 拿到股票列表，你必须:
  - 取出列表中前N只 (用户要几只就取几只，默认前5只)
  - 对每一只逐一调用上面全部6个工具
  - 不能只扫不查、不能只查第一只就停

所有工具调完后，按股票逐一汇总所有数据。严禁分析、评分、给建议。
"""

ANALYST_PROMPT = """你是 FinBrain 高级分析师。根据搜集到的财务数据，按FinBrain双策略评分。

多股票规则：如果数据中包含多只股票，逐一评分输出JSON数组 [{股票1},{股票2},...]，独立评分互不影响。

数据缺失处理：如果某项数据未提供或显示"未找到"/"error"，该项得分标N/A且依据写"数据缺失"，不影响其他项。

[评分维度，每项0-10分]

1. 盈利能力: ROE>20%得7分起,>50%满分。毛利率>40%加2分。净利率>20%加1分。
2. 成长性: 连续两季营收增速>30%满分,单季>20%得5分,负增长0分。
3. 财务健康: 资产负债率<30%满分,30-50%得7分,50-60%得5分,>60%得3分。经营现金流/净利润>0.8加2分。
4. 估值合理: PE<15满分,15-25得7分,25-40得5分,>40得3分。
5. 行业前景: AI/半导体/新能源>7分,传统龙头>5分,夕阳<3分。
6. 资金认可: 主力净流入>1亿得7分起,净流入量大得满分,净流出>1亿扣分。未在排名前5页找到 -> 数据缺失,标N/A。

[输出格式]
严格输出JSON，不要markdown代码块。缺失项: "得分": null, "依据": "数据缺失":
{
  "代码": "xxx", "名称": "xxx",
  "评分": {
    "盈利能力": {"得分": N, "依据": "ROE 72.75%,..."},
    "成长性":   {"得分": N, "依据": "营收+187%,..."},
    "财务健康": {"得分": N, "依据": "负债率30%,..."},
    "估值合理": {"得分": N, "依据": "PE 53倍,..."},
    "行业前景": {"得分": N, "依据": "AI光模块龙头"},
    "资金认可": {"得分": N, "依据": "净流入19亿"}
  },
  "亮点": [...], "风险": [...],
  "操作建议": "一句话", "止损": "条件",
  "结论": {"总评":"...","买入策略":"...","持有策略":"...","卖出条件":"...","预期收益":"...","持仓周期":"..."}
}
"""

# ============================================================
#  节点函数
# ============================================================

_data_collector_tools = [stock_price, stock_history, financial_statements,
                         valuation, industry_info, screen_stocks, fund_flow]

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
    from tools import format_report
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

_CHAT_TOOLS = [stock_price, stock_history]
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
    """判断请求类型: 'phantom' / 'analysis' / 'chat'"""
    phantom_triggers = ["妖股", "猎妖", "涨停", "打板", "短线爆发"]
    analysis_triggers = ["分析", "报告", "评分", "筛选", "扫描", "对比", "选股"]
    if any(t in user_input for t in phantom_triggers):
        return "phantom"
    if any(t in user_input for t in analysis_triggers):
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
