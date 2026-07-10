"""
FinBrain — LangGraph Agent 主程序
独立的财报分析 Agent，不依赖 Claude Code / MCP 协议。
使用 LangGraph 预置的 ReAct 循环，LLM 自主决定调用哪个工具。
"""

import json
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# 根据 .env 配置自动选择 LLM 提供商
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic / deepseek / openai

if LLM_PROVIDER == "deepseek":
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        max_tokens=4096,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
elif LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI
    LLM = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
else:  # anthropic
    from langchain_anthropic import ChatAnthropic
    LLM = ChatAnthropic(
        model="claude-sonnet-5",
        temperature=0,
        max_tokens=4096,
    )

# ============================================================
#  用 @tool 装饰器把 tools.py 的纯函数包成 LLM 可调用的工具
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
        result["data"] = result["data"][-10:]  # 只保留最近10条，节省token
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
def screen_stocks(max_pe: float = 30, max_pb: float = 5,
                  min_mktcap: float = 20, top_n: int = 30) -> str:
    """全市场扫描，返回PE从低到高排列的A股。
    参数: max_pe=市盈率上限, max_pb=市净率上限, min_mktcap=最低市值(亿), top_n=返回条数"""
    from tools import screen_stocks as do_screen
    result = do_screen(max_pe, max_pb, min_mktcap, top_n)
    if "text" in result:
        return result["text"]
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
#  创建 Agent
# ============================================================

SYSTEM_PROMPT = """你是一个专业的A股财报分析助手 FinBrain，采用趋势投资+财报狙击双策略。

[核心策略]
1. 行业景气优先：先判断个股所处行业是否景气，再分析个股基本面
2. 龙头聚焦：每个行业只盯最受益、资金最认可的那家公司
3. 回调买入：好公司等回调10%-15%再介入，不追高
4. 趋势持有：对了死拿，错了砍仓
5. 回避狂热：不选当前市场最热门、交易最拥挤的股票。当一只股票被所有人谈论时，超额收益已经消失

[财报季狙击策略]
关键窗口：1/4/7/10月（财报密集期）
核心信号：财报发布后次日开盘跳空高开5%以上且缺口不回补 -> 资金认可
买点：等高点回调10%-15%轻仓试错，止损，后续根据表现加仓

[量化基本面验证]
收入增速：连续两季同比>30% -> 真实需求
毛利率：逐季提升 -> 议价能力增强
合同负债：同比翻倍 -> 未来收入硬保障

[错杀白马捡漏法]
条件1：ROE长期保持20%以上
条件2：股价从高点至少跌30%，估值近五年低位
条件3：最新财报利润、收入仍保持增长 -> 确认为被拖累
买入：先买20%仓位，每跌5%加10%，越跌越买

[估值指标]
NVC（净营运资本）：流动资产-全部流动负债，若远大于市值 -> 清算价值低估
ROE+PB组合：ROE>20%且PB<5倍，科技股看PB历史百分位

[工具说明]
- stock_price: 查实时股价
- stock_history: 查历史K线
- financial_statements: 查三大报表
- valuation: 查ROE/毛利率/净利率/EPS/资产负债率
- industry_info: 查行业分类
- fund_flow: 查当日资金流向（主力净流入/流出/净额）
- screen_stocks: 全市场PE/PB/市值扫描

[输出格式]
不使用emoji，不使用markdown加粗。
表格列宽必须等宽对齐：先计算每列最长值，所有格子填充空格至相同宽度。
示例：
  代码     名称       PE     PB     市值(亿)
  000498   山东路桥   4.2    0.45   752934
  000001   平安银行   5.1    0.44   20356808
每个格子右侧至少空2格。数字右对齐，中文左对齐。
所有指标用具体数字，标注同比变化%。
结论置于末尾，一句概括。

[分析步骤]
1. 根据用户问题调用工具获取数据
2. 对比三年数据判断趋势（改善/恶化/稳定）
3. 按策略条件打分：ROE、营收增速、毛利率趋势、现金流质量、负债率
4. 给出操作建议（等待回调/轻仓试错/远离）
"""


def build_agent():
    """构建 FinBrain Agent"""
    tools = [stock_price, stock_history, financial_statements, valuation, industry_info, screen_stocks, fund_flow]

    return create_agent(
        LLM,
        tools,
        system_prompt=SYSTEM_PROMPT,
    )


# ============================================================
#  交互入口
# ============================================================

def ask(question: str, history: list = None) -> str:
    """向 Agent 提问，支持传入历史消息"""
    agent = build_agent()
    messages = (history or []) + [{"role": "user", "content": question}]
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("FinBrain Agent")
    print("Type 'quit' to exit, 'clear' to reset context")
    print()

    agent = build_agent()
    history = []  # 维护对话历史

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

            messages = history + [{"role": "user", "content": user_input}]
            result = agent.invoke({"messages": messages})
            reply = result["messages"][-1].content
            print(reply)
            print("-" * 60)

            # 将本轮对话存入历史
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nbye")
            break
        except Exception as e:
            print(f"[Error] {e}")
