"""
FinBrain Streamlit 前端 — 暗色主题
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 加载 .env（Streamlit 不会自动加载，必须在所有 os.getenv() 之前执行）
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), "..", "configs", ".env")
load_dotenv(_env_path, override=True)

import streamlit as st
import json, time
from langchain_core.callbacks import BaseCallbackHandler
import plotly.graph_objects as go

st.set_page_config(page_title="FinBrain", layout="wide", initial_sidebar_state="expanded")

# ---- 工具调用追踪器 ----
class ToolCallTracker(BaseCallbackHandler):
    """拦截 LangChain tool call 事件，记录到列表"""
    def __init__(self):
        self.records = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.records.append({
            "tool": tool_name,
            "input": str(input_str)[:120],
            "time": time.strftime("%H:%M:%S"),
            "status": "running",
        })

    def on_tool_end(self, output, **kwargs):
        if self.records:
            self.records[-1]["status"] = "done"
            self.records[-1]["output_preview"] = str(output)[:80]

    def on_tool_error(self, error, **kwargs):
        if self.records:
            self.records[-1]["status"] = "error"
            self.records[-1]["error"] = str(error)[:80]

# ========== Kimi 风格暗色主题（主色保持红色 #cc3333） ==========
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  .stApp {
      background: linear-gradient(135deg, #0a0a0a 0%, #111111 50%, #0d0d0d 100%) !important;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  }
  p, label, h1, h2, h3, h4, li {
      color: #e8e8e8 !important;
      font-family: 'Inter', sans-serif !important;
  }
  div, span {
      font-family: 'Inter', sans-serif !important;
  }
  h1, h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em !important; }

  /* 头部 */
  header { background: rgba(10,10,10,0.85) !important; backdrop-filter: blur(20px) !important; }
  header h1, header h2, header h3, header p, header span { color: #e8e8e8 !important; }

  /* 侧边栏固定：默认展开、禁止折叠、折叠按钮隐藏 */
  section[data-testid="stSidebar"] {
      background: rgba(12,12,12,0.92) !important;
      backdrop-filter: blur(24px) !important;
      border-right: 1px solid rgba(255,255,255,0.06) !important;
      width: 280px !important;
      min-width: 280px !important;
      max-width: 280px !important;
      transform: none !important;
      transition: none !important;
  }
  /* 隐藏侧边栏折叠/展开按钮，避免用户误点收起 */
  button[data-testid="stSidebarCollapseButton"],
  button[data-testid="stSidebarExpandButton"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapsedControl"] button,
  button[data-testid="baseButton-headerNoPadding"] {
      display: none !important;
      visibility: hidden !important;
      opacity: 0 !important;
      width: 0 !important;
      height: 0 !important;
      padding: 0 !important;
      margin: 0 !important;
      pointer-events: none !important;
  }
  /* 强制主内容区让出固定侧边栏宽度 */
  .stApp [data-testid="stAppViewContainer"] > .main {
      margin-left: 280px !important;
  }
  /* 当侧边栏被意外折叠时，强制重新展开（防某些版本行为） */
  section[data-testid="stSidebar"][aria-expanded="false"] {
      width: 280px !important;
      min-width: 280px !important;
      max-width: 280px !important;
  }

  /* 侧边栏内部文字颜色 */
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div,
  section[data-testid="stSidebar"] label {
      color: #a0a0a0 !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #e8e8e8 !important; }
  section[data-testid="stSidebar"] .stRadio label {
      padding: 8px 12px !important;
      border-radius: 10px !important;
      transition: all 0.2s ease !important;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {
      background: rgba(204,51,51,0.10) !important;
      color: #cc3333 !important;
  }
  section[data-testid="stSidebar"] .stRadio label span {
      color: #e0e0e0 !important;
  }

  /* 按钮 */
  button[kind="primary"] {
      background: linear-gradient(135deg, #cc3333 0%, #a82a2a 100%) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 10px !important;
      font-weight: 500 !important;
      box-shadow: 0 4px 15px rgba(204,51,51,0.25) !important;
      transition: all 0.2s ease !important;
  }
  button[kind="primary"]:hover {
      transform: translateY(-1px) !important;
      box-shadow: 0 6px 20px rgba(204,51,51,0.35) !important;
  }
  button[kind="secondary"] {
      background: rgba(255,255,255,0.06) !important;
      color: #e0e0e0 !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      border-radius: 10px !important;
      transition: all 0.2s ease !important;
  }
  button[kind="secondary"]:hover {
      background: rgba(255,255,255,0.10) !important;
      border-color: rgba(204,51,51,0.3) !important;
  }

  /* 输入框 */
  input, textarea, .stTextInput input, .stTextArea textarea {
      background: rgba(25,25,25,0.8) !important;
      color: #e8e8e8 !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      border-radius: 12px !important;
      font-family: 'Inter', sans-serif !important;
  }
  input:focus, textarea:focus, .stTextInput input:focus, .stTextArea textarea:focus {
      border-color: #cc3333 !important;
      box-shadow: 0 0 0 3px rgba(204,51,51,0.15) !important;
  }

  /* 聊天消息 */
  .stChatMessage {
      background: rgba(25,25,25,0.7) !important;
      backdrop-filter: blur(12px) !important;
      border-radius: 16px !important;
      border: 1px solid rgba(255,255,255,0.05) !important;
      margin: 10px 0 !important;
      padding: 16px !important;
  }
  .stChatMessage [data-testid="stChatMessageAvatar"] { color: #cc3333 !important; }
  .stChatMessage p, .stChatMessage span { color: #e0e0e0 !important; }

  /* Tabs */
  .stTabs [role="tablist"] {
      background: rgba(20,20,20,0.6) !important;
      border-radius: 12px !important;
      padding: 4px !important;
      border: 1px solid rgba(255,255,255,0.05) !important;
  }
  .stTabs button {
      background: transparent !important;
      border-radius: 8px !important;
      color: #999 !important;
      border: none !important;
  }
  .stTabs button[aria-selected="true"] {
      background: rgba(204,51,51,0.12) !important;
      color: #cc3333 !important;
      font-weight: 600 !important;
  }

  /* DataFrame / Table */
  .stDataFrame, .stTable, [data-testid="stDataFrameResizable"] {
      background: rgba(20,20,20,0.6) !important;
      border-radius: 12px !important;
      border: 1px solid rgba(255,255,255,0.05) !important;
  }
  .stDataFrame th, .stDataFrame td { color: #e0e0e0 !important; border-color: rgba(255,255,255,0.05) !important; }
  .stDataFrame th { background: rgba(30,30,30,0.8) !important; }

  /* 报告块 */
  .report-block {
      background: rgba(20,20,20,0.75) !important;
      backdrop-filter: blur(12px) !important;
      border-left: 3px solid #cc3333 !important;
      border-radius: 16px !important;
      padding: 24px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,0.25) !important;
  }
  .report-block pre {
      background: transparent !important;
      color: #e0e0e0 !important;
      font-family: 'JetBrains Mono', 'Consolas', monospace !important;
      font-size: 13px !important;
      line-height: 1.75 !important;
      white-space: pre-wrap !important;
      margin: 0 !important;
      padding: 0 !important;
      border: none !important;
  }

  /* 卡片 */
  .card {
      background: rgba(25,25,25,0.7) !important;
      backdrop-filter: blur(10px) !important;
      border: 1px solid rgba(255,255,255,0.06) !important;
      border-radius: 16px !important;
      padding: 20px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
      transition: transform 0.2s ease !important;
  }
  .card:hover { transform: translateY(-2px) !important; }
  .card .title { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
  .card .value { font-size: 32px; font-weight: 700; color: #cc3333; margin-top: 6px; }
  .card .value.green { color: #4caf50; }
  .card .sub { font-size: 12px; color: #777; margin-top: 4px; }

  /* Metric Box (Market page) */
  .metric-box {
      background: rgba(25,25,25,0.7) !important;
      backdrop-filter: blur(10px) !important;
      border: 1px solid rgba(255,255,255,0.06) !important;
      border-radius: 16px !important;
      padding: 18px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
  }
  .metric-box .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
  .metric-box .value { font-size: 28px; font-weight: 700; color: #e8e8e8; margin-top: 6px; }
  .metric-box .sub { font-size: 12px; color: #777; margin-top: 4px; }

  /* 选择器 / 下拉 */
  .stSelectbox div[data-baseweb="select"],
  .stMultiSelect div[data-baseweb="select"] {
      background: rgba(25,25,25,0.8) !important;
      border-radius: 12px !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
  }
  .stSelectbox div[data-baseweb="select"] > div,
  .stMultiSelect div[data-baseweb="select"] > div { color: #e8e8e8 !important; }

  /* Expander */
  .stExpander {
      background: rgba(20,20,20,0.6) !important;
      border-radius: 12px !important;
      border: 1px solid rgba(255,255,255,0.05) !important;
  }
  .stExpander summary { color: #e0e0e0 !important; font-weight: 500 !important; }
  .stExpander summary:hover { color: #cc3333 !important; }

  /* Streamlit 原生底部输入框统一风格 */
  [data-testid="stChatInput"] {
      background: rgba(25,25,25,0.9) !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      border-radius: 14px !important;
      height: 48px !important;
      font-size: 15px !important;
      /* 缩小宽度，为右侧模式切换按钮留空间 */
      width: calc(100% - 120px) !important;
  }
  [data-testid="stChatInput"] input:focus {
      border-color: #cc3333 !important;
      box-shadow: 0 0 0 3px rgba(204,51,51,0.15) !important;
  }
  /* 输入框右侧模式切换按钮，与输入框同高 */
  [data-testid="stPopover"] > button {
      background: linear-gradient(135deg, #cc3333 0%, #a82a2a 100%) !important;
      border: none !important;
      border-radius: 12px !important;
      height: 48px !important;
      width: 110px !important;
      color: #fff !important;
      font-weight: 600 !important;
      font-size: 14px !important;
      box-shadow: 0 4px 15px rgba(204,51,51,0.25) !important;
      /* 绝对定位到输入框右侧 */
      position: fixed !important;
      bottom: 16px !important;
      right: 24px !important;
      z-index: 10000 !important;
  }
  [data-testid="stPopover"] > button:hover {
      background: linear-gradient(135deg, #d94444 0%, #b83333 100%) !important;
  }
  /* 隐藏 popover 的默认容器位置 */
  [data-testid="stPopover"] > div:first-child {
      position: fixed !important;
      bottom: 0 !important;
      right: 0 !important;
      z-index: 10000 !important;
  }

  /* 信息框 */
  .stAlert {
      border-radius: 12px !important;
      border: 1px solid rgba(255,255,255,0.06) !important;
  }
  .stAlert [data-testid="stAlertContent"] { color: #e0e0e0 !important; }
  .stInfo { background: rgba(59,130,246,0.08) !important; border-left: 3px solid #3b82f6 !important; }
  .stSuccess { background: rgba(76,175,80,0.08) !important; border-left: 3px solid #4caf50 !important; }
  .stWarning { background: rgba(255,193,7,0.08) !important; border-left: 3px solid #ffc107 !important; }
  .stError { background: rgba(244,67,54,0.08) !important; border-left: 3px solid #f44336 !important; }

  /* 分割线 */
  hr {
      border-color: rgba(255,255,255,0.06) !important;
      margin: 24px 0 !important;
  }

  /* 滚动条 */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #0a0a0a; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #cc3333; }

  footer { visibility: hidden; }
  div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# 兜底：确保侧边栏始终固定展开（配合 CSS 隐藏折叠按钮）
st.markdown("""
<script>
(function() {
    function lockSidebar() {
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.setProperty('width', '280px', 'important');
            sidebar.style.setProperty('min-width', '280px', 'important');
            sidebar.style.setProperty('max-width', '280px', 'important');
            sidebar.style.setProperty('transform', 'none', 'important');
            sidebar.style.setProperty('transition', 'none', 'important');
            sidebar.setAttribute('aria-expanded', 'true');
        }
        var main = document.querySelector('.main');
        if (main) main.style.setProperty('margin-left', '280px', 'important');
        var toggles = document.querySelectorAll(
            'button[data-testid="stSidebarCollapseButton"], ' +
            'button[data-testid="stSidebarExpandButton"], ' +
            '[data-testid="stSidebarCollapsedControl"], ' +
            '[data-testid="stSidebarCollapsedControl"] button, ' +
            'button[data-testid="baseButton-headerNoPadding"]'
        );
        toggles.forEach(function(el) {
            el.style.setProperty('display', 'none', 'important');
            el.style.setProperty('visibility', 'hidden', 'important');
            el.style.setProperty('opacity', '0', 'important');
            el.style.setProperty('pointer-events', 'none', 'important');
        });
    }
    lockSidebar();
    setTimeout(lockSidebar, 300);
    setTimeout(lockSidebar, 1000);
})();
</script>
""", unsafe_allow_html=True)

# Logo 区域
st.markdown("""
<style>
  .kimi-logo {
      display: flex; align-items: center; gap: 12px;
      padding: 12px 0 20px 0; margin-bottom: 8px;
  }
  .kimi-logo .logo-mark {
      width: 32px; height: 32px; border-radius: 10px;
      background: linear-gradient(135deg, #cc3333 0%, #a82a2a 100%);
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-weight: 700; font-size: 16px; font-family: 'Inter', sans-serif;
      box-shadow: 0 4px 12px rgba(204,51,51,0.3);
  }
  .kimi-logo .logo-text { font-size: 20px; font-weight: 700; color: #e8e8e8; letter-spacing: -0.02em; }
  .kimi-logo .logo-sub { font-size: 12px; color: #777; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown('''
    <div class="kimi-logo">
        <div class="logo-mark">F</div>
        <div>
            <div class="logo-text">FinBrain</div>
            <div class="logo-sub">AI-Powered Research</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["Market", "Chat", "Portfolio", "Analysis", "Knowledge", "Evaluation", "Backtest", "Settings"], label_visibility="collapsed")
    st.divider()
    # 会话持久化（LangGraph SqliteSaver + Streamlit session_state）
    import uuid
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"

    st.button("Clear", on_click=lambda: [
        st.session_state.chat_history.clear(),
        st.session_state.pop("thread_id", None)
    ])
    st.divider()
    st.caption("FinBrain v0.3")


# ========== 公共 ==========
def get_agents():
    from backend.agent import build_graph, _get_chat_agent, _get_phantom_agent, _classify_request
    return {"graph": build_graph(), "chat": _get_chat_agent(),
            "phantom": _get_phantom_agent(), "classify": _classify_request}

def _to_lc(h):
    from langchain_core.messages import HumanMessage, AIMessage
    return [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in h]

class StreamHandler(BaseCallbackHandler):
    """捕获LLM流式输出的每个token，更新Streamlit占位符"""
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.tokens = ""
    def on_llm_new_token(self, token, **kwargs):
        self.tokens += token
        self.placeholder.text(self.tokens)


def run_agent(user_input: str, stream_placeholder=None) -> tuple[str, list, str, list]:
    """返回 (回复文本, 工具调用记录列表, 流式生成过程文本, 执行Trace)。
    Trace格式: [{"phase": "Data", "summary": "...", "detail": "..."}, ...]
    """
    agents = get_agents()
    # 历史压缩：超过阈值时用LLM摘要旧消息，只保留最近N条原文
    from backend.agent import compress_history
    compressed = compress_history(st.session_state.chat_history)
    msgs = _to_lc(compressed) + [{"role": "user", "content": user_input}]
    tracker = ToolCallTracker()
    callbacks = [tracker]
    stream_handler = None
    if stream_placeholder is not None:
        stream_handler = StreamHandler(stream_placeholder)
        callbacks.append(stream_handler)
    cfg = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "callbacks": callbacks,
    }

    mode = st.session_state.get("mode", "Chat")
    if mode == "Chat":
        auto = agents["classify"](user_input)
        if auto == "analysis": mode = "Deep Analysis"
        elif auto == "phantom": mode = "Phantom Hunter"

    trace = []
    if mode == "Phantom Hunter":
        reply = agents["phantom"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content
        trace = [{"phase": "Phantom", "summary": "妖股猎人生成完成", "detail": reply[:500]}]
    elif mode == "Deep Analysis":
        r = agents["graph"].invoke({"messages": msgs, "user_question": user_input,
                                     "collected_data":"", "analysis":"", "report":"",
                                     "processing_log": []}, config=cfg)
        reply = r.get("report")
        if not reply:
            raw = r.get("analysis","")
            if raw.strip():
                reply = f"[流水线未生成报告，以下是原始分析摘要]\n\n{raw[:2000]}"
            else:
                reply = "[流水线执行失败，请重试或切换到闲聊模式]"
        trace = r.get("processing_log", [])
    else:
        reply = agents["chat"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content
        trace = [{"phase": "Chat", "summary": "ReAct对话完成", "detail": reply[:500]}]

    # 追加工具调用痕迹到回复（Deep Analysis 报告已有 [调用证据]，仅 Chat/Phantom 追加）
    if mode != "Deep Analysis" and tracker.records:
        tool_names = list(set(r.get("tool", "?") for r in tracker.records[:20]))
        if tool_names:
            reply += f"\n\n[调用证据] 工具: {', '.join(tool_names)}"
    elif mode == "Deep Analysis" and tracker.records:
        # Deep Analysis 报告内已有 [调用证据]，但如果缺失则补上
        if "[调用证据]" not in str(reply):
            tool_names = list(set(r.get("tool", "?") for r in tracker.records[:20]))
            if tool_names:
                reply += f"\n\n[调用证据] 工具: {', '.join(tool_names)}"

    stream_text = stream_handler.tokens if stream_handler else ""
    return reply, tracker.records, stream_text, trace


# ========== Market ==========
if page == "Market":
    st.header("Market Monitor")

    # ---- 刷新 ----
    if st.button("Refresh Data", type="primary"):
        st.rerun()

    # ---- 涨跌全景 (缓存5分钟，避免每次rerun都调API) ----
    @st.cache_data(ttl=300)
    def _cached_breadth():
        from backend.tools import get_market_breadth
        return get_market_breadth()
    breadth = _cached_breadth()
    if "error" not in breadth:
        a = breadth["全A"]
        up_pct = a["上涨"] / a["总计"] * 100 if a["总计"] else 0
        dn_pct = a["下跌"] / a["总计"] * 100 if a["总计"] else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="label">Up</div><div class="value">{a["上涨"]}</div><div class="sub">{up_pct:.0f}%</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="label">Down</div><div class="value">{a["下跌"]}</div><div class="sub">{dn_pct:.0f}%</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="label">Flat</div><div class="value">{a["平盘"]}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="label">Breadth</div><div class="value">{breadth["上涨比例"]}</div></div>', unsafe_allow_html=True)
    st.divider()

    # ---- 中线动量聚焦 (缓存5分钟) ----
    @st.cache_data(ttl=300)
    def _cached_momentum():
        from backend.tools import get_sector_momentum
        return get_sector_momentum(15)

    momentum = _cached_momentum()
    mom_list = momentum.get("列表", [])
    if mom_list and "error" not in momentum:
        st.subheader(f"中线动量聚焦 {momentum.get('市场情绪','')}")
        st.caption("追踪主力共识最强的板块——'趋势中继'而非'底部反转'。加速期适合关注，高潮期只出不进。")
        cols = st.columns(min(len(mom_list[:10]), 5))
        temp_colors = {"🔥高潮期": "#cc3333", "⚡加速期": "#e69500", "🌡️升温中": "#4a90d9", "❄️观望": "#666"}
        for i, m in enumerate(mom_list[:10]):
            with cols[i % 5]:
                tc = temp_colors.get(m["温度计"], "#666")
                st.markdown(
                    f'<div style="border-left:3px solid {tc}; padding:4px 8px; margin:2px 0; font-size:13px">'
                    f'<b>{i+1}. {m["板块"]}</b> <span style="color:{tc}">{m["温度计"]}</span><br>'
                    f'<span style="font-size:11px;color:#aaa">动量{m["动量分数"]:.0f} | {m["净流入(亿)"]:+.1f}亿 | {m["涨跌幅"]}</span><br>'
                    f'<span style="font-size:10px;color:#888">{m["逻辑"]}</span></div>',
                    unsafe_allow_html=True
                )
    st.divider()

    # ---- 板块资金流对比图 (缓存5分钟) ----
    @st.cache_data(ttl=300)
    def _cached_sector_total():
        from backend.tools import get_sector_fund_flow
        return get_sector_fund_flow(100, fund_type="total")
    @st.cache_data(ttl=300)
    def _cached_sector_main():
        from backend.tools import get_sector_fund_flow
        return get_sector_fund_flow(100, fund_type="main")

    sector_data = _cached_sector_total()
    # 主力资金仅个股页面有，行业页面只有全市场——暂时只用全市场数据
    chart_mode = "全市场"  # 未来: 接入个股主力数据后可切换

    def _draw_sector_chart(data, title_prefix, key_suffix):
        all_sectors = data.get("列表", [])
        if not all_sectors:
            st.warning("暂无数据")
            return
        sort_by = st.radio("排序", ["净流入额", "涨跌幅"], horizontal=True, key=f"sector_sort_{key_suffix}")
        if sort_by == "涨跌幅":
            all_sectors.sort(key=lambda s: float(str(s.get("涨跌幅", "0%")).replace("%","").replace("+","") or 0), reverse=True)
        else:
            all_sectors.sort(key=lambda s: abs(s["净额(亿)"]), reverse=True)
        names = [s["板块"] for s in all_sectors]
        nets = [s["净额(亿)"] for s in all_sectors]
        changes = [float(str(s.get("涨跌幅", "0%")).replace("%","").replace("+","") or 0) for s in all_sectors]
        colors = ["#cc3333" if n >= 0 else "#2e7d32" for n in nets]
        if sort_by == "涨跌幅":
            bar_values, x_title = changes, "涨跌幅 (%)"
        else:
            bar_values, x_title = [abs(n) for n in nets], "|净流入| (亿)"
        bar_text = [f"{n:+.2f}亿  {c:+.2f}%" for n, c in zip(nets, changes)]
        fig = go.Figure(data=[go.Bar(x=bar_values, y=names, orientation='h', marker_color=colors,
                                      text=bar_text, textposition='outside', textfont=dict(color='#ddd', size=9))])
        fund_label = data.get("资金类型", "")
        update_time = data.get("更新时间", "")
        data_date = data.get("数据日期", "")
        title_text = f"{title_prefix}{fund_label} ({len(all_sectors)} sectors)"
        if update_time:
            title_text += f"<br><sup>数据日期: {data_date} | 更新时间: {update_time}</sup>"
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=14, color="#e8e8e8", family="Inter, sans-serif")),
            height=max(600, len(names)*20),
            margin=dict(l=10, r=100, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b0b0b0", family="Inter, sans-serif"),
            xaxis=dict(title=x_title, showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#888")),
            yaxis=dict(showgrid=False, tickfont=dict(color="#b0b0b0")),
            hoverlabel=dict(bgcolor="rgba(20,20,20,0.9)", font_color="#e8e8e8", bordercolor="rgba(255,255,255,0.1)")
        )
        st.plotly_chart(fig, use_container_width=True, key=f"kline_{key_suffix}")

    _draw_sector_chart(sector_data, "", "total")

    # 下方表格
    all_sec = sector_data.get("列表", [])
    with st.expander(f"Sector Fund Flow Details ({len(all_sec)} sectors)", expanded=False):
        lines = [f"{'Sector':<12} {'Chg':>8} {'In(亿)':>10} {'Out(亿)':>10} {'Net(亿)':>10}"]
        lines.append("-" * 55)
        for s in all_sec:
            net = s["净额(亿)"]
            color = "#cc3333" if net >= 0 else "#2e7d32"
            lines.append(
                f"<span style='color:{color}'>{s['板块']:<12} {s['涨跌幅']:>8} "
                f"{s['流入(亿)']:>10.2f} {s['流出(亿)']:>10.2f} {net:>+10.2f}</span>"
            )
        st.markdown("<pre style='font-size:12px'>" + "\n".join(lines) + "</pre>",
                    unsafe_allow_html=True)
# ========== Chat ==========
if page == "Chat":
    st.header("AI Chat")

    # ---- 对话历史 ----
    history = st.session_state.chat_history
    last_stream = st.session_state.get("_last_stream", "")
    last_trace = st.session_state.get("_last_trace", [])
    for i, msg in enumerate(history):
        is_last = (i == len(history) - 1)
        is_assistant = (msg["role"] == "assistant")
        with st.chat_message(msg["role"]):
            # 最后一条 assistant 消息：结构化 Execution Trace
            if is_last and is_assistant and last_trace:
                with st.expander("Execution Trace", expanded=False):
                    _icons = {"Data": "📡", "Classify": "🏷️", "Analysis": "🧠", "Valuation": "📊", "Critics": "🔍", "Repair": "🔧", "Report": "📝"}
                    _status_colors = {"SUCCESS": "#2e7d32", "WARNING": "#e69500", "PARTIAL": "#cc6600"}
                    for step in last_trace:
                        phase = step.get("phase", "?")
                        icon = _icons.get(phase, "⚙️")
                        status = step.get("status", "SUCCESS")
                        sc = _status_colors.get(status, "#666")
                        summary = step.get("summary", "")
                        detail = step.get("detail", "")
                        # Header
                        st.markdown(f"**{icon} {phase}** — <span style='color:{sc}'>{status}</span> | {summary}", unsafe_allow_html=True)
                        # Rich details per phase
                        if phase == "Data":
                            lat = step.get("latency_ms", 0)
                            syms = step.get("symbols", [])
                            errs = step.get("errors", 0)
                            actions = step.get("actions", [])
                            st.caption(f"Latency: {lat}ms | Symbols: {len(syms)} | Errors: {errs}")
                            if actions:
                                _done = [a for a in actions if a["status"] == "✅"]
                                _fail = [a for a in actions if a["status"] != "✅"]
                                st.caption("Actions: " + ", ".join(f"{'✅' if a['status']=='✅' else '❌'}{a['tool']}" for a in actions[:10]))
                        elif phase == "Analysis":
                            chars = step.get("output_chars", 0)
                            rags = step.get("rag_calls", [])
                            st.caption(f"Output: {chars} chars | RAG: {'; '.join(rags[:3]) if rags else '无'}")
                        elif phase == "Valuation":
                            stage = step.get("stage", "?")
                            frameworks = step.get("frameworks", [])
                            ref = step.get("reference", {})
                            st.caption(f"Stage: {stage} | Frameworks: {', '.join(frameworks[:4])}")
                            if ref:
                                st.caption("Reference: " + " | ".join(f"{k}:{v}" for k, v in list(ref.items())[:3]))
                        elif phase == "Critics":
                            findings = step.get("findings", {})
                            conf = step.get("confidence", "?")
                            decision = step.get("decision", "?")
                            st.caption(f"Findings: 逻辑{findings.get('逻辑漏洞',0)} | 过度乐观{findings.get('过度乐观',0)} | 遗漏风险{findings.get('遗漏风险',0)} | 置信度:{conf} | 决策:{decision}")
                            if detail:
                                st.caption(detail.replace("\n", " | ")[:250])
                        elif phase == "Repair":
                            fix_count = step.get("fix_count", 0)
                            st.caption(f"Fixed: {fix_count} issues auto-corrected")
                        elif phase == "Report":
                            chars = step.get("output_chars", 0)
                            retries = step.get("audit_retries", 0)
                            max_r = step.get("max_retries", 3)
                            precheck = step.get("code_precheck", "?")
                            st.caption(f"Output: {chars} chars | Audit: {retries}/{max_r} retries | Code precheck: {precheck}")
                        st.divider()
            if is_last and is_assistant and last_stream:
                with st.expander("Generation trace", expanded=False):
                    st.caption(last_stream[:10000] if len(last_stream) > 10000 else last_stream)
            # 报告：每行最多75字，超出自动换行
            content = msg["content"]
            if is_assistant and ("====" in content[:200] or "[投资决策]" in content[:500]):
                wrapped = []
                for line in content.split("\n"):
                    while len(line) > 75:
                        wrapped.append(line[:75])
                        line = line[75:]
                    wrapped.append(line)
                st.code("\n".join(wrapped), language=None, line_numbers=False)
            else:
                st.text(content)

    # ---- 底部留白（防止内容被固定输入栏遮挡） ----
    st.markdown('<div style="height:90px"></div>', unsafe_allow_html=True)

    # ---- 输入行：Streamlit 原生固定底部输入框 + 右侧模式切换 ----
    current_mode = st.session_state.get("mode", "Chat")
    _MODE_META = {
        "Chat":           {"icon": "💬", "label": "闲聊"},
        "Deep Analysis":  {"icon": "📊", "label": "分析"},
        "Phantom Hunter": {"icon": "🔮", "label": "妖股"},
    }
    cur = _MODE_META[current_mode]

    # st.chat_input 自动固定在页面底部；用 CSS 缩小宽度，右侧放模式切换
    prompt = st.chat_input("Ask FinBrain...")

    # 模式切换：放在 chat_input 右侧，用 CSS 绝对定位
    with st.popover(f"{cur['icon']} {cur['label']}", use_container_width=False):
        for mode, meta in _MODE_META.items():
            bt = "primary" if current_mode == mode else "secondary"
            if st.button(f"{meta['icon']} {meta['label']}", use_container_width=True, type=bt, key=f"pop_mode_{mode}"):
                st.session_state.mode = mode
                st.rerun()

    if prompt:
        with st.chat_message("user"): st.text(prompt)

        with st.chat_message("assistant"):
            with st.expander("Generating... (streaming)", expanded=True):
                stream_box = st.empty()
            stream_text = ""
            trace = []
            try:
                reply, tool_logs, stream_text, trace = run_agent(prompt, stream_placeholder=stream_box)
            except Exception as e:
                reply = f"[Error] {e}"
                stream_box.text(reply)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state["_last_stream"] = stream_text  # 保存流式生成过程
        st.session_state["_last_trace"] = trace  # 保存执行Trace
        st.rerun()


# ========== Portfolio ==========
elif page == "Portfolio":
    st.header("Portfolio Management")
    from backend.portfolio import get_portfolio, list_accounts, delete_account
    accounts = list_accounts()
    if not accounts:
        get_portfolio("default")  # 自动创建默认账户
        accounts = list_accounts()
    acc_names = [a["name"] for a in accounts]
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: cur_acc = st.selectbox("账户", acc_names, key="pf_account")
    with c2:
        new_name = st.text_input("新建", placeholder="账户名", key="pf_new", label_visibility="collapsed")
    with c3:
        if st.button("创建", key="pf_create") and new_name.strip():
            get_portfolio(new_name.strip())
            st.rerun()
        if st.button("删除", key="pf_del") and cur_acc and len(acc_names) > 1:
            delete_account(cur_acc)
            st.rerun()
    pf = get_portfolio(cur_acc); d = pf.summary()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="card"><div class="title">Cash</div><div class="value">{d["现金"]:,.0f}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="card"><div class="title">Positions</div><div class="value">{d["持仓市值"]:,.0f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="card"><div class="title">Total Assets</div><div class="value">{d["总资产"]:,.0f}</div><div class="sub">Return {d["累计收益率"]}</div></div>', unsafe_allow_html=True)
    with c4:
        pnl = d["总盈亏"]; cls = "green" if pnl > 0 else ""
        st.markdown(f'<div class="card"><div class="title">PnL</div><div class="value {cls}">{pnl:,.0f}</div><div class="sub">{d["总盈亏%"]}</div></div>', unsafe_allow_html=True)
    st.divider()
    if d["持仓明细"]:
        st.subheader("Positions")
        st.dataframe(d["持仓明细"], use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")
    st.divider()
    st.subheader("Place Order")
    c_a,c_b,c_c,c_d = st.columns(4)
    with c_a: action = st.selectbox("Action", ["buy","sell"])
    with c_b: sym = st.text_input("Symbol", placeholder="300502")
    with c_c: pct = st.number_input("Pct %", 1, 100, 5)
    with c_d:
        st.write("");st.write("")
        if st.button("Execute Order", type="primary", use_container_width=True):
            if not sym: st.warning("Enter stock code")
            elif action=="buy":
                r=pf.buy_pct(sym,pct)
                st.success(f"{r.get('name',sym)} x{r.get('shares',0)} cost {r.get('cost',0):,.0f}") if "error" not in r else st.error(r["error"])
            else:
                r=pf.sell_pct(sym,pct)
                st.success(f"Sold {sym}: {r.get('pnl_pct','')}") if "error" not in r else st.error(r["error"])
    st.divider()
    with st.popover("Reset Portfolio"):
        nc = st.number_input("Initial Capital", value=1_000_000, step=100_000)
        if st.button("Reset", type="primary"): pf.reset(nc); st.rerun()


# ========== Analysis ==========
elif page == "Analysis":
    st.header("Stock Analysis")

    # ---- 搜索：代码或名称 ----
    query = st.text_input("Stock Code or Name", placeholder="e.g. 300502 or 新易盛", key="analysis_query")

    # ---- 名称→代码解析 ----
    resolved_code = None
    resolved_name = None
    if query and query.strip():
        q = query.strip()
        if q.isdigit() and len(q) == 6:
            resolved_code = q
            from backend.stock_map import code_to_name
            resolved_name = code_to_name(q) or q
        else:
            from backend.stock_map import fuzzy_search
            results = fuzzy_search(q, limit=5)
            if results:
                options = [f"{r['代码']} {r['名称']}" for r in results]
                selected = st.selectbox("匹配结果", options, key="analysis_match")
                if selected:
                    resolved_code = selected.split()[0]
                    resolved_name = selected.split(maxsplit=1)[1] if len(selected.split()) > 1 else selected
            else:
                st.warning(f"未找到匹配 '{q}' 的股票")

    # ---- K线图 ----
    if resolved_code:
        st.caption(f"**{resolved_name}** ({resolved_code})")

        c_tf, _ = st.columns([1.5, 3])
        with c_tf:
            timeframe = st.selectbox("", ["分时", "五日", "日K", "周K", "月K"], key="analysis_tf")

        from frontend.kline_chart import build_kline_chart
        try:
            fig = build_kline_chart(resolved_code, resolved_name, timeframe)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"kline_{resolved_code}")
            else:
                st.warning("暂无数据（非交易时段分时数据可能为空）")
        except Exception as e:
            st.error(f"Chart error: {e}")

        # ---- 财报分析按钮 + Web Search 交叉验证 ----
        st.divider()
        use_web = st.checkbox("Web Search 交叉验证 (PE/PB/市值/目标价)", value=False,
                              help="勾选后用配置的搜索API验证容易过时的估值数据。不勾选则先用LLM自带的web search功能，失败后沿用API数据。")

        if st.button("Deep Analysis (财报+估值)", type="primary", use_container_width=True):
            # 构建分析指令
            if use_web:
                instruction = (f"分析{resolved_code}的财报和估值。"
                               f"[!!!] 对PE/PB/市值/目标价/买入区间等容易过时的数据，"
                               f"必须调用 web_search 工具搜索 '{resolved_name} {resolved_code} PE PB 市值 最新' 进行交叉验证。"
                               f"如果web_search结果与API数据差异>15%，以web_search为准。"
                               f"财报数据(营收/利润/ROE/毛利率)仍使用 financial_statements 和 valuation 工具。")
            else:
                instruction = (f"分析{resolved_code}的财报和估值。"
                               f"PE/PB/市值等估值数据优先使用LLM自带的web search能力验证。"
                               f"如果无法联网搜索，沿用API数据和自算结果。"
                               f"财报数据仍使用 financial_statements 和 valuation 工具。")
            with st.spinner(f"Analyzing {resolved_name}... (20-40s)"):
                try:
                    reply, _, _, trace = run_agent(instruction)
                    if trace:
                        with st.expander("Execution Trace", expanded=False):
                            _icons = {"Data": "📡", "Classify": "🏷️", "Analysis": "🧠", "Valuation": "📊", "Critics": "🔍", "Repair": "🔧", "Report": "📝"}
                            for step in trace:
                                icon = _icons.get(step.get("phase", "?"), "⚙️")
                                status = step.get("status", "SUCCESS")
                                summary = step.get("summary", "")
                                st.caption(f"{icon} **{step.get('phase','?')}** [{status}]: {summary}")
                                if step.get("phase") == "Data":
                                    st.caption(f"  Latency: {step.get('latency_ms',0)}ms | Errors: {step.get('errors',0)}")
                                elif step.get("phase") == "Critic":
                                    f = step.get("findings", {})
                                    st.caption(f"  逻辑:{f.get('逻辑漏洞',0)} 过度乐观:{f.get('过度乐观',0)} 遗漏:{f.get('遗漏风险',0)} | 置信度:{step.get('confidence','?')}")
                                elif step.get("phase") == "Report":
                                    st.caption(f"  Audit: {step.get('audit_retries',0)}/{step.get('max_retries',3)} retries | Precheck: {step.get('code_precheck','?')}")
                    wrapped = []
                    for line in reply.split("\n"):
                        while len(line) > 75:
                            wrapped.append(line[:75])
                            line = line[75:]
                        wrapped.append(line)
                    st.html(f'<div class="report-block"><pre>{"\n".join(wrapped)}</pre></div>')
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check API Key in Settings and restart.")

    elif not query:
        st.info("Enter a stock code or name to run deep analysis")


# ========== Knowledge ==========
elif page == "Knowledge":
    st.header("Knowledge Base")

    from backend.accounting_rag import (
        list_kbs, list_documents, upload_document, delete_document,
        search_kb, get_kb_stats, seed_accounting_kb,
    )

    # ---- 确保预置知识已播种 ----
    if "kb_seeded" not in st.session_state:
        try:
            result = seed_accounting_kb()
            st.session_state.kb_seeded = True
            if result["seeded"] > 0:
                st.toast(f"已初始化会计准则知识库 ({result['seeded']}条)", icon="✅")
        except Exception as e:
            st.session_state.kb_seeded = True  # 不重试
            st.warning(f"预置知识初始化: {e}")

    # ---- 知识库列表 (缓存30秒，避免每次rerun查询ChromaDB) ----
    @st.cache_data(ttl=30)
    def _cached_kbs():
        try:
            return list_kbs()
        except Exception:
            return []
    kbs = _cached_kbs()

    kb_names = [k["name"] for k in kbs] if kbs else ["accounting"]
    kb_labels = {k["name"]: f"{k['display_name']} ({k['doc_count']} docs)" for k in kbs} if kbs else {"accounting": "会计准则"}

    kb_tab1, kb_tab2, kb_tab3 = st.tabs(["Upload", "Manage", "Search"])

    # -- Tab1: Upload --
    with kb_tab1:
        st.subheader("Upload Document")
        target_kb = st.selectbox(
            "Target Knowledge Base", kb_names,
            format_func=lambda n: kb_labels.get(n, n),
            key="kb_upload_select",
        )
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "txt", "md", "csv", "json"],
            key="kb_file_uploader",
            help="支持 PDF/DOCX/TXT/MD/CSV/JSON，自动解析+切片+向量化",
        )
        if uploaded_file is not None:
            if st.button("Index Document", type="primary", key="kb_index_btn"):
                with st.spinner(f"Parsing & indexing '{uploaded_file.name}'..."):
                    try:
                        result = upload_document(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                            target_kb,
                        )
                        st.success(
                            f"Indexed: **{result['filename']}** → "
                            f"{result['chunks']} chunks ({result['size']:,} bytes)"
                        )
                        with st.expander("Preview (first 300 chars)", expanded=False):
                            st.caption(result.get("preview", "")[:300])
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

    # -- Tab2: Manage --
    with kb_tab2:
        view_kb = st.selectbox(
            "Knowledge Base", kb_names,
            format_func=lambda n: kb_labels.get(n, n),
            key="kb_manage_select",
        )
        try:
            docs = list_documents(view_kb)
            stats = get_kb_stats(view_kb)
            st.caption(f"{stats['total_documents']} docs, {stats['total_chunks']} chunks, {stats['total_chars']:,} chars")

            if docs:
                for d in docs:
                    is_builtin = d.get("source") == "builtin"
                    badge = "🔒内置" if is_builtin else "📄"
                    c1, c2 = st.columns([6, 1])
                    with c1:
                        st.caption(
                            f"{badge} **{d['filename']}** — "
                            f"{d['total_chunks']} chunks · {d['total_chars']} chars · {d['upload_date']}"
                        )
                    with c2:
                        if not is_builtin and st.button("Delete", key=f"del_{d['doc_id']}"):
                            if delete_document(d['doc_id'], view_kb):
                                st.success("Deleted")
                                st.rerun()
                            else:
                                st.error("Failed")
            else:
                st.info(f"No documents in '{view_kb}'")
        except Exception as e:
            st.error(f"Failed to load documents: {e}")

    # -- Tab3: Search --
    with kb_tab3:
        st.subheader("Semantic Search")
        search_kb_sel = st.selectbox(
            "Knowledge Base", kb_names,
            format_func=lambda n: kb_labels.get(n, n),
            key="kb_search_select",
        )
        search_query = st.text_input("Query", placeholder="e.g. 收入确认五步法 / 商誉减值测试条件", key="kb_search_input")

        if search_query and st.button("Search", type="primary", key="kb_search_btn"):
            with st.spinner("Searching..."):
                try:
                    results = search_kb(search_query, search_kb_sel, top_k=5)
                    if results:
                        for i, r in enumerate(results):
                            score_color = "#4caf50" if r["score"] > 0.7 else ("#ff9800" if r["score"] > 0.4 else "#cc3333")
                            st.markdown(
                                f"**#{i+1}** `{r['source']}` [{r['chunk']}] "
                                f"<span style='color:{score_color}'>score: {r['score']}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(r["content"][:400] + ("..." if len(r["content"]) > 400 else ""))
                            st.divider()
                    else:
                        st.info(f"No results found in '{search_kb_sel}'")
                except Exception as e:
                    st.error(f"Search failed: {e}")

# ========== Evaluation ==========
elif page == "Evaluation":
    st.header("Agent Evaluation")
    st.caption("量化评估 Agent 输出的稳定性、完整性和可靠性。不评估投资建议准确性——那是回测的事。")

    c1, c2 = st.columns([2, 1])
    with c1:
        n_stocks = st.number_input("测试股票数量", min_value=1, max_value=10, value=2, key="eval_n")
    with c2:
        n_runs = st.number_input("每只运行次数", min_value=1, max_value=100, value=5, key="eval_runs")

    symbols = []
    cols = st.columns(min(n_stocks, 5))
    for i in range(n_stocks):
        with cols[i % 5]:
            s = st.text_input(f"股票{i+1}", placeholder="600584", key=f"eval_sym_{i}")
            if s.strip(): symbols.append(s.strip())

    if st.button("Run Evaluation", type="primary", disabled=len(symbols)==0, use_container_width=True):
        from backend.evaluation import evaluate_stock
        results = []
        progress = st.progress(0)
        total = len(symbols)
        for idx, sym in enumerate(symbols):
            st.caption(f"Evaluating {sym} ({n_runs} runs)...")
            r = evaluate_stock(sym, int(n_runs))
            results.append(r)
            progress.progress((idx + 1) / total)
        progress.empty()

        if results:
            st.divider()
            st.subheader("Results")
            # Summary table
            st.dataframe(
                [{k: v for k, v in r.items() if k != "原始分数"} for r in results],
                use_container_width=True, hide_index=True,
                column_config={
                    "代码": st.column_config.TextColumn("Stock", width="small"),
                    "评分一致性": st.column_config.ProgressColumn("Score Consistency", format="%.0f%%", min_value=0, max_value=100),
                    "字段完整率": st.column_config.ProgressColumn("Field Completeness", format="%.0f%%", min_value=0, max_value=100),
                    "工具成功率": st.column_config.ProgressColumn("Tool Success", format="%.0f%%", min_value=0, max_value=100),
                    "评分标准差": st.column_config.NumberColumn("Score StdDev", format="%.2f"),
                    "平均延迟_ms": st.column_config.NumberColumn("Avg Latency(ms)", format="%.0f"),
                }
            )
            # Overall metrics
            avg_cons = sum(r["评分一致性"] for r in results) / len(results)
            avg_field = sum(r["字段完整率"] for r in results) / len(results)
            avg_tool = sum(r["工具成功率"] for r in results) / len(results)
            total_errs = sum(r["错误次数"] for r in results)
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Score Consistency", f"{avg_cons:.0f}%")
            with c2: st.metric("Field Completeness", f"{avg_field:.0f}%")
            with c3: st.metric("Tool Success", f"{avg_tool:.0f}%")
            with c4: st.metric("Total Errors", total_errs)
            # Score distribution
            with st.expander("Score Distribution", expanded=False):
                for r in results:
                    if r.get("原始分数"):
                        st.caption(f"{r['代码']}: {r['原始分数']}")

# ========== Backtest ==========
elif page == "Backtest":
    st.header("Backtest Engine")
    st.caption("用历史K线回测分析报告的交易建议。纯代码，不调LLM。")

    from backend.backtest import extract_signal, run_backtest, batch_backtest
    from backend.portfolio import list_accounts as _list_accts, delete_account as _del_acct, Portfolio

    tab1, tab2 = st.tabs(["单报告回测", "批量回测 + 账户管理"])

    with tab1:
        st.subheader("粘贴报告文本")
        report_text = st.text_area("报告内容", height=300, placeholder="粘贴 FinBrain 生成的完整报告...",
                                   key="bt_report")
        lookback = st.slider("回看天数", 30, 365, 180, key="bt_lookback")
        if st.button("Run Backtest", type="primary", key="bt_run") and report_text.strip():
            signal = extract_signal(report_text)
            if signal:
                st.json(signal)
                result = run_backtest(signal, lookback)
                if "error" in result:
                    st.error(result["error"])
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("触发", "✅" if result["triggered"] else "❌")
                    with c2: st.metric("收益率", f"{result['return_pct']:+.1f}%")
                    with c3: st.metric("持仓天数", result["holding_days"])
                    st.caption(f"入场: {result['entry_date']} @ {result['entry_price']} | "
                               f"出场: {result['exit_date']} @ {result['exit_price']} | "
                               f"原因: {result['exit_reason']}")
            else:
                st.warning("未能从报告中提取交易信号（建仓价/止损/目标价）")

    with tab2:
        st.subheader("批量回测")
        reports_input = st.text_area("多份报告（每份用---分隔）", height=200,
                                     placeholder="报告1\n---\n报告2\n---\n报告3",
                                     key="bt_batch")
        if st.button("Batch Backtest", type="primary", key="bt_batch_run") and reports_input.strip():
            reports = [r.strip() for r in reports_input.split("---") if r.strip()]
            summary = batch_backtest(reports, lookback)
            if "error" in summary:
                st.warning(summary["error"])
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("总信号", summary["总信号数"])
                with c2: st.metric("触发率", f"{summary['触发数']}/{summary['总信号数']}")
                with c3: st.metric("胜率", f"{summary['胜率']}%")
                with c4: st.metric("平均收益", f"{summary['平均收益']:+.1f}%")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("最大收益", f"{summary['最大收益']:+.1f}%")
                with c2: st.metric("最大亏损", f"{summary['最大亏损']:+.1f}%")
                with c3: st.metric("盈亏比", f"{summary['胜数']}:{summary['败数']}")
                with st.expander("明细", expanded=False):
                    st.dataframe(summary["明细"], use_container_width=True)

        st.divider()
        st.subheader("模拟账户管理")
        accounts = _list_accts()
        acc_names = [a["name"] for a in accounts]
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            new_acc = st.text_input("新建账户", placeholder="value_strategy", key="bt_new_acc")
        with c2:
            if st.button("创建", key="bt_create") and new_acc.strip():
                Portfolio(new_acc.strip())
                st.rerun()
        with c3:
            del_acc = st.selectbox("删除", [""] + acc_names, key="bt_del_acc")
            if st.button("确认删除", key="bt_del_btn") and del_acc:
                _del_acct(del_acc)
                st.rerun()

        if accounts:
            st.caption(f"共 {len(accounts)} 个账户")
            for a in accounts:
                tv = a.get("total_value", a.get("cash", 0))
                pnl = tv - a.get("initial_cash", tv)
                st.caption(f"{a['name']}: 现金{a.get('cash',0):,.0f} | "
                           f"持仓{a.get('positions',0)}只 | 总资产{tv:,.0f} | "
                           f"盈亏{pnl:+,.0f}")

# ========== Settings ==========
elif page == "Settings":
    st.header("Settings")
    tab0,tab1,tab2,tab3,tab4 = st.tabs(["Strategy","LLM Provider","Tools","Data Sources","System"])
    with tab0:
        import json as _json
        strat_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "strategies.json")
        if os.path.exists(strat_file):
            with open(strat_file, encoding="utf-8") as f: strategies = _json.load(f)
        else:
            strategies = {"default":{"name":"Default"}}

        strategy_names = list(strategies.keys())
        cur_strat = os.getenv("FINBRAIN_STRATEGY","default")
        cur_idx = strategy_names.index(cur_strat) if cur_strat in strategy_names else 0

        selected = st.selectbox("Active Strategy", strategy_names, index=cur_idx,
                                format_func=lambda k: strategies[k].get("name",k))
        st.caption(strategies[selected].get("description",""))

        if selected != cur_strat and st.button("Activate Strategy", type="primary"):
            os.environ["FINBRAIN_STRATEGY"] = selected
            # 写入 .env
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if not line.startswith("FINBRAIN_STRATEGY="):
                            lines.append(line.rstrip())
            lines.append(f"FINBRAIN_STRATEGY={selected}")
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            st.success(f"Strategy switched to: {strategies[selected]['name']}. Restart to apply.")
            st.info("Run: streamlit run frontend/app.py")

        st.divider()
        st.subheader("Add / Edit Strategy")

        with st.form("strategy_form"):
            new_key = st.text_input("Strategy Key (英文ID)", placeholder="my_strategy")
            new_name = st.text_input("Strategy Name", placeholder="My Custom Strategy")
            new_desc = st.text_input("Description", placeholder="Brief description")
            new_trigger_a = st.text_input("Analysis Triggers (comma separated)", value="分析,报告")
            new_trigger_p = st.text_input("Phantom Triggers (comma separated)", value="妖股,涨停")
            new_collector = st.text_area("Data Collector Prompt", height=120,
                                         placeholder="数据搜集专员的system prompt...")
            new_analyst = st.text_area("Analyst Prompt", height=200,
                                       placeholder="分析师的system prompt...")
            new_phantom = st.text_area("Phantom Hunter Prompt", height=120,
                                       placeholder="妖股猎人的system prompt...")

            c_save, c_del = st.columns(2)
            with c_save:
                if st.form_submit_button("Save Strategy", type="primary"):
                    if not new_key or not new_name:
                        st.error("Strategy Key and Name are required")
                    else:
                        strategies[new_key] = {
                            "name": new_name,
                            "description": new_desc,
                            "triggers": {
                                "analysis": [t.strip() for t in new_trigger_a.split(",") if t.strip()],
                                "phantom": [t.strip() for t in new_trigger_p.split(",") if t.strip()]
                            },
                            "data_collector": new_collector,
                            "analyst": new_analyst,
                            "phantom": new_phantom
                        }
                        with open(strat_file, "w", encoding="utf-8") as f:
                            _json.dump(strategies, f, ensure_ascii=False, indent=2)
                        st.success(f"Strategy '{new_key}' saved. Restart to use.")
            with c_del:
                if selected != "default" and st.form_submit_button("Delete Selected"):
                    del strategies[selected]
                    with open(strat_file, "w", encoding="utf-8") as f:
                        _json.dump(strategies, f, ensure_ascii=False, indent=2)
                    os.environ["FINBRAIN_STRATEGY"] = "default"
                    st.success(f"Deleted '{selected}'. Reset to default. Restart.")

    with tab1:
        st.subheader("LLM Fallback Chain")
        st.caption("Slot 1 必填；Slot 2/3 为可选熔断备用。当 Slot 1 调用失败时，系统自动依次尝试下一槽位。")

        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
        providers = ["deepseek", "openai", "anthropic"]
        model_defaults = {"deepseek": "deepseek-chat", "openai": "gpt-4o", "anthropic": "claude-sonnet-5"}
        base_defaults = {"deepseek": "https://api.deepseek.com", "openai": "", "anthropic": ""}

        def _read_slot(i: int) -> dict:
            prefix = f"LLM_SLOT_{i}"
            provider = os.getenv(f"{prefix}_PROVIDER", "").strip()
            model = os.getenv(f"{prefix}_MODEL", "").strip()
            api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
            base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
            # 兼容旧单变量：仅 slot 1
            if i == 1 and not provider and not model:
                provider = os.getenv("LLM_PROVIDER", "deepseek").strip()
                model = os.getenv("LLM_MODEL", "").strip() or model_defaults.get(provider, "")
                old_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
                api_key = api_key or old_key
                base_url = base_url or os.getenv("LLM_BASE_URL", base_defaults.get(provider, ""))
            return {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }

        slots = []
        for i in range(1, 4):
            cur = _read_slot(i)
            required = i == 1
            enabled = required or st.checkbox(
                f"启用 Slot {i}", value=cur["provider"] != "",
                key=f"llm_slot_{i}_enabled", disabled=required
            )
            with st.expander(f"Slot {i} {'*' if required else '(optional)'} {'✅' if enabled else '❌'}", expanded=enabled):
                provider = st.selectbox(
                    "Provider",
                    providers,
                    index=providers.index(cur["provider"]) if cur["provider"] in providers else 0,
                    key=f"llm_slot_{i}_provider",
                    disabled=not enabled,
                )
                model = st.text_input(
                    "Model",
                    value=cur["model"] or model_defaults.get(provider, ""),
                    placeholder=model_defaults.get(provider, ""),
                    key=f"llm_slot_{i}_model",
                    disabled=not enabled,
                )
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=cur["api_key"],
                    placeholder="sk-...",
                    key=f"llm_slot_{i}_apikey",
                    disabled=not enabled,
                )
                base_url = st.text_input(
                    "Base URL (optional)",
                    value=cur["base_url"] or base_defaults.get(provider, ""),
                    placeholder=base_defaults.get(provider, ""),
                    key=f"llm_slot_{i}_baseurl",
                    disabled=not enabled,
                )
                slots.append({
                    "provider": provider,
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "required": required,
                    "enabled": enabled,
                })

        if st.button("Apply & Save", type="primary"):
            # 校验 slot 1
            if not slots[0]["provider"] or not slots[0]["model"]:
                st.error("Slot 1 必须填写 Provider 和 Model")
            else:
                # 写环境变量（只写入启用的 slot；禁用 slot 2/3 时清理变量）
                for i, s in enumerate(slots, start=1):
                    prefix = f"LLM_SLOT_{i}"
                    if not s["enabled"]:
                        for suffix in ["_PROVIDER", "_MODEL", "_API_KEY", "_BASE_URL"]:
                            key = f"{prefix}{suffix}"
                            if key in os.environ:
                                del os.environ[key]
                        continue
                    os.environ[f"{prefix}_PROVIDER"] = s["provider"]
                    os.environ[f"{prefix}_MODEL"] = s["model"]
                    if s["api_key"]:
                        os.environ[f"{prefix}_API_KEY"] = s["api_key"]
                    if s["base_url"]:
                        os.environ[f"{prefix}_BASE_URL"] = s["base_url"]
                    # 如果留空，删除旧环境变量
                    if not s["api_key"] and f"{prefix}_API_KEY" in os.environ:
                        del os.environ[f"{prefix}_API_KEY"]
                    if not s["base_url"] and f"{prefix}_BASE_URL" in os.environ:
                        del os.environ[f"{prefix}_BASE_URL"]

                # 重写 .env：清理旧单变量 + 写入 slot 变量
                lines = []
                if os.path.exists(env_path):
                    with open(env_path, encoding="utf-8") as f:
                        for line in f:
                            if any(line.startswith(p) for p in [
                                "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL",
                                "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"
                            ]):
                                continue
                            # 清理已有的 LLM_SLOT_* 行，后面会重新写入
                            if line.startswith("LLM_SLOT_"):
                                continue
                            lines.append(line.rstrip())

                for i, s in enumerate(slots, start=1):
                    if not s["enabled"]:
                        continue
                    prefix = f"LLM_SLOT_{i}"
                    lines.append(f"{prefix}_PROVIDER={s['provider']}")
                    lines.append(f"{prefix}_MODEL={s['model']}")
                    if s["api_key"]: lines.append(f"{prefix}_API_KEY={s['api_key']}")
                    if s["base_url"]: lines.append(f"{prefix}_BASE_URL={s['base_url']}")

                with open(env_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

                configured = [i for i, s in enumerate(slots, start=1) if s["enabled"] and s["provider"] and s["model"]]
                st.success(f"Saved {len(configured)} slot(s): {', '.join(f'Slot {i}' for i in configured)}. Restart to apply.")

    with tab2:
        st.caption("10 tools active")
        for t in ["stock_price","stock_history","financial_statements","valuation",
                  "industry_info","screen_stocks","fund_flow","limit_up_pool","concept_ranking","dragon_tiger_list"]:
            st.checkbox(t, value=True, key=f"tool_{t}")

    with tab3:
        st.subheader("Data Sources")
        st.caption("默认使用免费数据源。可自定义替换为付费或私有源。修改后需重启生效。")

        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")

        # 预设定义
        _PRESETS = {
            "free": {"name": "免费源(默认)", "stock_price": "sina", "financials": "eastmoney",
                     "industry": "eastmoney_ths", "fund_flow": "ths"},
            "akshare": {"name": "AkShare", "stock_price": "akshare", "financials": "akshare",
                        "industry": "akshare", "fund_flow": "akshare"},
            "custom": {"name": "自定义", "stock_price": "", "financials": "", "industry": "", "fund_flow": ""},
        }

        # 读取当前值
        def _read_env(key, default):
            return os.getenv(key, default)

        cur_stock = _read_env("DATA_SOURCE_STOCK_PRICE", "sina")
        cur_fin = _read_env("DATA_SOURCE_FINANCIALS", "eastmoney")
        cur_ind = _read_env("DATA_SOURCE_INDUSTRY", "eastmoney_ths")
        cur_fund = _read_env("DATA_SOURCE_FUND_FLOW", "ths")

        # 判断当前预设
        cur_preset = "custom"
        for pk, pv in _PRESETS.items():
            if pk == "custom": continue
            if (cur_stock == pv["stock_price"] and cur_fin == pv["financials"]
                and cur_ind == pv["industry"] and cur_fund == pv["fund_flow"]):
                cur_preset = pk; break

        preset = st.selectbox("Preset", list(_PRESETS.keys()),
                              index=list(_PRESETS.keys()).index(cur_preset),
                              format_func=lambda k: _PRESETS[k]["name"])

        # 根据预设填充
        if preset != "custom":
            p = _PRESETS[preset]
            cur_stock, cur_fin, cur_ind, cur_fund = p["stock_price"], p["financials"], p["industry"], p["fund_flow"]

        with st.form("datasource_form"):
            c1, c2 = st.columns(2)
            with c1:
                ds_stock = st.text_input("Stock Price Source", value=cur_stock,
                                         placeholder="sina", help="股价数据源ID")
                ds_fin = st.text_input("Financial Statements Source", value=cur_fin,
                                       placeholder="eastmoney", help="财报数据源ID")
            with c2:
                ds_ind = st.text_input("Industry Info Source", value=cur_ind,
                                       placeholder="eastmoney_ths", help="行业分类数据源ID")
                ds_fund = st.text_input("Fund Flow Source", value=cur_fund,
                                        placeholder="ths", help="资金流向数据源ID")

            if st.form_submit_button("Save & Apply", type="primary"):
                # 校验非空
                if not all([ds_stock, ds_fin, ds_ind, ds_fund]):
                    st.error("所有数据源字段不能为空。请填写有效的数据源ID或使用预设。")
                else:
                    try:
                        # 写入 .env
                        env_vars = {
                            "DATA_SOURCE_STOCK_PRICE": ds_stock,
                            "DATA_SOURCE_FINANCIALS": ds_fin,
                            "DATA_SOURCE_INDUSTRY": ds_ind,
                            "DATA_SOURCE_FUND_FLOW": ds_fund,
                        }
                        lines = []
                        if os.path.exists(env_path):
                            with open(env_path, encoding="utf-8") as f:
                                for line in f:
                                    keep = True
                                    for k in env_vars:
                                        if line.startswith(k + "="):
                                            keep = False; break
                                    if keep: lines.append(line.rstrip())
                        for k, v in env_vars.items():
                            lines.append(f"{k}={v}")
                            os.environ[k] = v
                        with open(env_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines) + "\n")
                        st.success("Data sources saved. Restart to apply changes.")
                        st.info("Run: streamlit run frontend/app.py")
                    except Exception as e:
                        st.error(f"写入配置失败: {e}")

        st.divider()
        st.subheader("Web Search API")
        st.caption("用于交叉验证免费API数据。默认使用 Tavily Search（tvly-xxx 格式 key）。")

        cur_ws_provider = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
        cur_ws_key = os.getenv("WEB_SEARCH_API_KEY", "")
        cur_ws_url = os.getenv("WEB_SEARCH_BASE_URL", "")

        ws_provider = st.selectbox("Provider", ["tavily", "serpapi", "serper", "custom"],
                                   index=["tavily","serpapi","serper","custom"].index(cur_ws_provider)
                                   if cur_ws_provider in ["tavily","serpapi","serper","custom"] else 0,
                                   key="ws_provider")
        ws_key = st.text_input("API Key", type="password", value=cur_ws_key,
                               placeholder="tvly-xxx (Tavily)", key="ws_key")
        if ws_provider == "custom":
            ws_url = st.text_input("Base URL", value=cur_ws_url, placeholder="https://api.example.com/search", key="ws_url")
        else:
            ws_url = ""

        c_ws1, c_ws2 = st.columns(2)
        with c_ws1:
            if st.button("Save Web Search Config", type="primary", key="save_ws"):
                if not ws_key.strip():
                    st.error("API Key 不能为空")
                else:
                    try:
                        # 智能 key 识别
                        from backend.web_search import _detect_key_type, llm_detect_key_type
                        detected = _detect_key_type(ws_key.strip())
                        if not detected and ws_provider == "tavily" and not ws_key.strip().startswith("tvly-"):
                            with st.spinner("正在识别 API Key 类型..."):
                                llm_r = llm_detect_key_type(ws_key.strip())
                                detected = llm_r.get("provider", "custom")
                                st.info(f"LLM 识别结果: {llm_r.get('provider')} — {llm_r.get('reason', '')}")

                        # 写入 .env
                        ws_vars = {"WEB_SEARCH_PROVIDER": ws_provider, "WEB_SEARCH_API_KEY": ws_key.strip()}
                        if ws_url.strip():
                            ws_vars["WEB_SEARCH_BASE_URL"] = ws_url.strip()
                        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
                        lines = []
                        if os.path.exists(env_path):
                            with open(env_path, encoding="utf-8") as f:
                                for line in f:
                                    if not any(line.startswith(k + "=") for k in ws_vars):
                                        lines.append(line.rstrip())
                        for k, v in ws_vars.items():
                            lines.append(f"{k}={v}")
                            os.environ[k] = v
                        with open(env_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines) + "\n")
                        st.success(f"Web Search config saved. Provider: {ws_provider}. Restart to apply.")
                    except Exception as e:
                        st.error(f"保存失败: {e}")
        with c_ws2:
            if cur_ws_key:
                st.caption(f"当前: {cur_ws_provider} | Key: {cur_ws_key[:8]}...{cur_ws_key[-4:]}")
            else:
                st.caption("未配置 (非财报数据不会交叉验证)")

        st.divider()
        st.subheader("Data API Slots")
        st.caption("Slot 1/2/3 为可选高级数据源。免费 API 失败时系统按顺序尝试高级插槽。Slot 1 不强制。")

        data_providers = ["none", "tushare", "wind", "choice", "ifind", "bloomberg"]

        def _read_data_slot(i: int) -> dict:
            prefix = f"DATA_SLOT_{i}"
            return {
                "provider": os.getenv(f"{prefix}_PROVIDER", "").strip().lower(),
                "api_key": os.getenv(f"{prefix}_API_KEY", "").strip(),
                "base_url": os.getenv(f"{prefix}_BASE_URL", "").strip(),
                "extra": os.getenv(f"{prefix}_EXTRA", "").strip(),
            }

        data_slots = []
        for i in range(1, 4):
            cur = _read_data_slot(i)
            enabled = cur["provider"] != "" and cur["provider"] != "none"
            with st.expander(f"Data Slot {i} {'✅' if enabled else '❌'}", expanded=enabled):
                provider = st.selectbox(
                    "Provider",
                    data_providers,
                    index=data_providers.index(cur["provider"]) if cur["provider"] in data_providers else 0,
                    key=f"data_slot_{i}_provider",
                )
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=cur["api_key"],
                    key=f"data_slot_{i}_apikey",
                )
                base_url = st.text_input(
                    "Base URL (optional)",
                    value=cur["base_url"],
                    key=f"data_slot_{i}_baseurl",
                )
                extra = st.text_input(
                    "Extra params (optional)",
                    value=cur["extra"],
                    key=f"data_slot_{i}_extra",
                )
                data_slots.append({
                    "provider": provider,
                    "api_key": api_key,
                    "base_url": base_url,
                    "extra": extra,
                })

        c_save, c_test = st.columns([1, 1])
        with c_save:
            if st.button("Save Data API Slots", type="primary", key="save_data_slots"):
                try:
                    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
                    lines = []
                    if os.path.exists(env_path):
                        with open(env_path, encoding="utf-8") as f:
                            for line in f:
                                if not line.startswith("DATA_SLOT_"):
                                    lines.append(line.rstrip())
                    for i, s in enumerate(data_slots, start=1):
                        prefix = f"DATA_SLOT_{i}"
                        if s["provider"] and s["provider"] != "none":
                            lines.append(f"{prefix}_PROVIDER={s['provider']}")
                            if s["api_key"]:
                                lines.append(f"{prefix}_API_KEY={s['api_key']}")
                            if s["base_url"]:
                                lines.append(f"{prefix}_BASE_URL={s['base_url']}")
                            if s["extra"]:
                                lines.append(f"{prefix}_EXTRA={s['extra']}")
                            os.environ[f"{prefix}_PROVIDER"] = s["provider"]
                            os.environ[f"{prefix}_API_KEY"] = s["api_key"]
                            os.environ[f"{prefix}_BASE_URL"] = s["base_url"]
                            os.environ[f"{prefix}_EXTRA"] = s["extra"]
                        else:
                            for suffix in ["_PROVIDER", "_API_KEY", "_BASE_URL", "_EXTRA"]:
                                key = f"{prefix}{suffix}"
                                if key in os.environ:
                                    del os.environ[key]
                    with open(env_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    st.success("Data API Slots saved. Restart to apply.")
                except Exception as e:
                    st.error(f"保存失败: {e}")

        with c_test:
            if st.button("Test Tushare", key="test_tushare"):
                try:
                    from backend.data_slots import TushareProvider
                    cfg = {
                        "api_key": os.getenv("DATA_SLOT_1_API_KEY", ""),
                        "base_url": os.getenv("DATA_SLOT_1_BASE_URL", ""),
                        "extra": os.getenv("DATA_SLOT_1_EXTRA", ""),
                    }
                    p = TushareProvider(cfg)
                    if p.connected:
                        st.success("Tushare API Key 可连接")
                    else:
                        st.error("Tushare 连接失败，请检查 Slot 1 的 API Key")
                except Exception as e:
                    st.error(f"Tushare 测试失败: {e}")

        st.divider()
        st.caption("当前数据源状态:")
        st.code(f"Stock Price:  {cur_stock}\nFinancials:   {cur_fin}\nIndustry:     {cur_ind}\nFund Flow:    {cur_fund}\nWeb Search:   {cur_ws_provider} ({'已配置' if cur_ws_key else '未配置'})\nData Slots:   {', '.join(s['provider'] for s in [_read_data_slot(i) for i in range(1,4)] if s['provider'] and s['provider'] != 'none') or '未配置'}", language=None)

    with tab4:
        st.subheader("Deployment Mode")
        cur_data_mode = os.getenv("FINBRAIN_DATA_MODE", "local")
        cur_llm_mode = os.getenv("FINBRAIN_LLM_MODE", "local")
        cur_api_url = os.getenv("FINBRAIN_DATA_API", "http://localhost:8000")

        dm = st.selectbox("Data Mode", ["local", "remote"],
                          index=0 if cur_data_mode == "local" else 1,
                          help="local=本地直连数据源 / remote=调远程 Data API")
        lm = st.selectbox("LLM Mode", ["local", "remote_client"],
                          index=0 if cur_llm_mode == "local" else 1,
                          help="local=服务端调LLM / remote_client=客户端本地调LLM(Key不上传)")
        api_url = st.text_input("Data API URL", value=cur_api_url,
                                placeholder="http://your-server:8000",
                                disabled=(dm == "local"))

        status = "Local" if dm == "local" else f"Remote ({api_url})"
        if dm == "remote":
            try:
                import urllib.request, json
                r = urllib.request.urlopen(f"{api_url}/health", timeout=3)
                h = json.loads(r.read())
                status = f"Remote Connected ({api_url}) - v{h.get('version','?')}"
            except Exception:
                status = f"Remote Offline ({api_url})"

        st.caption(f"Status: **{status}** | LLM: **{lm}**")
        if dm == "remote_client":
            st.info("远程LLM模式: Prompt由服务器组装, 推理在你本地执行。你的API Key不会上传。")

        if st.button("Save Deployment Config", type="primary", key="save_deploy"):
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        if not any(line.startswith(p) for p in
                                   ["FINBRAIN_DATA_MODE","FINBRAIN_LLM_MODE","FINBRAIN_DATA_API"]):
                            lines.append(line.rstrip())
            lines.append(f"FINBRAIN_DATA_MODE={dm}")
            lines.append(f"FINBRAIN_LLM_MODE={lm}")
            if api_url.strip(): lines.append(f"FINBRAIN_DATA_API={api_url.strip()}")
            os.environ["FINBRAIN_DATA_MODE"] = dm
            os.environ["FINBRAIN_LLM_MODE"] = lm
            os.environ["FINBRAIN_DATA_API"] = api_url.strip()
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            st.success("Saved. Restart to apply.")

        st.divider()
        c1,c2 = st.columns(2)
        with c1: st.number_input("Compress Trigger", value=int(os.getenv("COMPRESS_TRIGGER","12")), min_value=4, key="ct2")
        with c2: st.number_input("Compress Keep", value=int(os.getenv("COMPRESS_KEEP","6")), min_value=2, key="ck2")
        if st.button("Save Compress Config", type="primary", key="save_compress"):
            os.environ["COMPRESS_TRIGGER"] = str(st.session_state.get("ct2",12))
            os.environ["COMPRESS_KEEP"] = str(st.session_state.get("ck2",6))
            st.success("Applied")
    st.divider(); st.caption("FinBrain v0.3")
