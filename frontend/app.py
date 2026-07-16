"""
FinBrain Streamlit 前端 — 暗色主题
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import json, time
from langchain_core.callbacks import BaseCallbackHandler
import plotly.graph_objects as go

st.set_page_config(page_title="FinBrain", layout="wide")

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

# ========== 暗色主题 ==========
st.markdown("""
<style>
  .stApp { background: #111; }
  p, span, div, label, h1, h2, h3, h4, li { color: #ddd !important; }
  header { background: #1a1a1a !important; }
  header * { color: #ccc !important; }
  section[data-testid="stSidebar"] { background: #1a1a1a !important; }
  section[data-testid="stSidebar"] * { color: #bbb !important; }
  section[data-testid="stSidebar"] h3 { color: #cc3333 !important; }
  button { background: #cc3333 !important; color: #fff !important; border: none !important; }
  input, textarea { background: #222 !important; color: #ddd !important; border: 1px solid #444 !important; }
  .stChatMessage { background: #1a1a1a !important; }
  .stTabs [role="tablist"], .stTabs button { background: #1a1a1a !important; }
  .stTabs button[aria-selected="true"] { border-bottom: 2px solid #cc3333 !important; }
  .stDataFrame, .stTable { background: #1a1a1a !important; }
  .report-block { background: #1a1a1a; border-left: 3px solid #cc3333; padding: 20px;
                  border-radius: 4px; }
  .report-block pre { background: transparent; color: #ddd; font-family: Consolas,monospace;
                      font-size: 13px; line-height: 1.7; white-space: pre-wrap;
                      margin: 0; padding: 0; border: none; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 20px; }
  .card .title { font-size: 13px; color: #888; }
  .card .value { font-size: 28px; font-weight: 700; color: #cc3333; }
  .card .value.green { color: #4caf50; }
  .card .sub { font-size: 12px; color: #999; }
  footer { visibility: hidden; }
  div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("### FinBrain")
    st.caption("AI-Powered Investment Research")
    st.divider()
    page = st.radio("", ["Market", "Chat", "Portfolio", "Analysis", "Knowledge", "Settings"], label_visibility="collapsed")
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
    st.caption("2026 FinBrain v0.2")


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


def run_agent(user_input: str, stream_placeholder=None) -> tuple[str, list]:
    """返回 (回复文本, 工具调用记录列表)。
    如果 stream_placeholder 不为空，流式输出到该占位符。
    """
    agents = get_agents()
    msgs = _to_lc(st.session_state.chat_history) + [{"role": "user", "content": user_input}]
    tracker = ToolCallTracker()
    callbacks = [tracker]
    if stream_placeholder is not None:
        callbacks.append(StreamHandler(stream_placeholder))
    cfg = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "callbacks": callbacks,
    }

    mode = st.session_state.get("mode", "Chat")
    if mode == "Chat":
        auto = agents["classify"](user_input)
        if auto == "analysis": mode = "Deep Analysis"
        elif auto == "phantom": mode = "Phantom Hunter"

    if mode == "Phantom Hunter":
        reply = agents["phantom"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content
    elif mode == "Deep Analysis":
        r = agents["graph"].invoke({"messages": msgs, "user_question": user_input,
                                     "collected_data":"", "analysis":"", "report":"",
                                     "processing_log": []}, config=cfg)
        reply = r.get("report") or r["messages"][-1].content
        proc_log = r.get("processing_log", [])
        if proc_log and stream_placeholder is None:
            with st.expander("Pipeline: Data -> Analysis -> Report", expanded=False):
                for step in proc_log:
                    phase = step.get("phase","?")
                    summary = step.get("summary","")
                    detail = step.get("detail","")
                    st.caption(f"[{phase}] {summary}")
                    if detail:
                        with st.expander(f"  {phase} detail", expanded=False):
                            st.text(detail[:1000])
    else:
        reply = agents["chat"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content

    return reply, tracker.records


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
        fig.update_layout(
            title=f"{title_prefix}{fund_label} ({len(all_sectors)} sectors)", height=max(600, len(names)*20),
            margin=dict(l=10, r=100, t=40, b=10), paper_bgcolor="#111", plot_bgcolor="#111",
            font=dict(color="#ddd"), xaxis=dict(title=x_title, showgrid=True, gridcolor="#333"), yaxis=dict(showgrid=False))
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

    # ---- 对话历史（最后的 assistant 消息如果是流式输出的，放在可折叠区内） ----
    history = st.session_state.chat_history
    last_stream = st.session_state.get("_last_stream", "")
    for i, msg in enumerate(history):
        is_last = (i == len(history) - 1)
        is_assistant = (msg["role"] == "assistant")
        is_streamed = is_last and is_assistant and last_stream and msg["content"] == last_stream
        with st.chat_message(msg["role"]):
            if is_streamed:
                with st.expander("Response (click to expand)", expanded=True):
                    st.text(msg["content"])
            else:
                st.text(msg["content"])

    # ---- 底部留白（防止内容被固定输入栏遮挡） ----
    st.markdown('<div style="height:90px"></div>', unsafe_allow_html=True)

    # ---- 输入行：固定于画面最下方 ----
    current_mode = st.session_state.get("mode", "Chat")
    _MODE_META = {
        "Chat":           {"icon": "💬", "label": "闲聊"},
        "Deep Analysis":  {"icon": "📊", "label": "分析"},
        "Phantom Hunter": {"icon": "🔮", "label": "妖股"},
    }
    cur = _MODE_META[current_mode]

    def _chat_send():
        """发送回调：设置待处理消息，输入框由 clear_on_submit 自动清空"""
        prompt = st.session_state.get("chat_text_input", "").strip()
        if not prompt:
            return
        st.session_state["_pending_chat"] = prompt

    # JS：把底部栏从 Streamlit 嵌套容器中移到 body 级别，实现真正的 fixed 定位
    st.markdown("""
    <style>
    #chat-bottom-bar {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: #111;
        padding: 12px 20px 16px 20px;
        z-index: 9999;
        border-top: 1px solid #333;
        transition: left 0.2s;
    }
    </style>
    <div id="chat-bottom-bar">
    """, unsafe_allow_html=True)

    ci, cm = st.columns([7.8, 1.6])
    with ci:
        with st.form("chat_form", clear_on_submit=True):
            st.text_input("Message", placeholder="Ask FinBrain... (Enter to send)",
                          label_visibility="collapsed", key="chat_text_input")
            if st.form_submit_button("➤"):
                _chat_send()
                st.rerun()
    with cm:
        popover_label = f"{cur['icon']} {cur['label']}"
        with st.popover(popover_label, use_container_width=True):
            st.caption("选择模式")
            for mode, meta in _MODE_META.items():
                bt = "primary" if current_mode == mode else "secondary"
                if st.button(f"{meta['icon']} {meta['label']}",
                             use_container_width=True, type=bt, key=f"pop_mode_{mode}"):
                    st.session_state.mode = mode
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # JS：把底部栏移到 body 级别固定定位
    st.markdown("""
    <script>
    (function() {
        var bar = document.getElementById('chat-bottom-bar');
        if (!bar || bar.parentElement === document.body) return;
        document.body.appendChild(bar);
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) bar.style.left = sidebar.offsetWidth + 'px';
        // 只在侧边栏折叠/展开时更新一次，不用高频observer
        var observer = new MutationObserver(function(mutations) {
            for (var m of mutations) {
                if (m.target.getAttribute('aria-expanded') !== null || m.target.className.indexOf('collapsed') >= 0) {
                    if (sidebar) bar.style.left = sidebar.offsetWidth + 'px';
                }
            }
        });
        var stApp = document.querySelector('.stApp');
        if (stApp) observer.observe(stApp, {attributes:true, subtree:true, attributeFilter:['class','aria-expanded']});
    })();
    </script>
    """, unsafe_allow_html=True)

    # ---- 处理待发送消息 ----
    if st.session_state.get("_pending_chat"):
        prompt = st.session_state.pop("_pending_chat")

        with st.chat_message("user"): st.text(prompt)

        with st.chat_message("assistant"):
            with st.expander("Generating... (streaming)", expanded=True):
                stream_box = st.empty()
            try:
                reply, tool_logs = run_agent(prompt, stream_placeholder=stream_box)
            except Exception as e:
                reply = f"[Error] {e}"
                stream_box.text(reply)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state["_last_stream"] = reply  # 保存流式文本供折叠查看
        st.rerun()


# ========== Portfolio ==========
elif page == "Portfolio":
    st.header("Portfolio Management")
    from backend.portfolio import get_portfolio
    pf = get_portfolio(); d = pf.summary()
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
                    reply, _ = run_agent(instruction)
                    st.html(f'<div class="report-block"><pre>{reply}</pre></div>')
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
        # 读取当前配置
        cur_provider = os.getenv("LLM_PROVIDER","deepseek")
        cur_model = os.getenv("LLM_MODEL", "")
        cur_key = os.getenv("DEEPSEEK_API_KEY","") or os.getenv("OPENAI_API_KEY","") or os.getenv("ANTHROPIC_API_KEY","")

        provider = st.selectbox("Provider", ["deepseek","openai","anthropic"],
                                index=["deepseek","openai","anthropic"].index(cur_provider) if cur_provider in ["deepseek","openai","anthropic"] else 0)
        # 模型默认值按 provider
        model_defaults = {"deepseek":"deepseek-chat","openai":"gpt-4o","anthropic":"claude-sonnet-5"}
        model = st.text_input("Model", value=cur_model or model_defaults.get(provider,""), placeholder=model_defaults.get(provider,""), key="settings_model")
        api_key = st.text_input("API Key", type="password", value=cur_key, placeholder="sk-...", key="settings_apikey")
        base_url = st.text_input("Base URL (optional)", value=os.getenv("LLM_BASE_URL","https://api.deepseek.com" if provider=="deepseek" else ""), key="settings_url")

        if st.button("Apply & Save", type="primary"):
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_MODEL"] = model
            if api_key:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
            if base_url:
                os.environ["LLM_BASE_URL"] = base_url
            # 写入 .env
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if not any(line.startswith(p) for p in ["LLM_PROVIDER","LLM_MODEL","LLM_BASE_URL",
                                "DEEPSEEK_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY"]):
                            lines.append(line.rstrip())
            lines.append(f"LLM_PROVIDER={provider}")
            lines.append(f"LLM_MODEL={model}")
            if api_key: lines.append(f"{provider.upper()}_API_KEY={api_key}")
            if base_url: lines.append(f"LLM_BASE_URL={base_url}")
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            st.success(f"Saved: {provider}/{model}. Restart to apply.")
                f.write(f"COMPRESS_TRIGGER={os.getenv('COMPRESS_TRIGGER','12')}\n")
                f.write(f"COMPRESS_KEEP={os.getenv('COMPRESS_KEEP','6')}\n")
            st.success(f"Saved. Key stored in configs/.env. Restart to apply new key.")
            st.info("Key change requires restart: `streamlit run frontend/app.py`")

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
        st.caption("当前数据源状态:")
        st.code(f"Stock Price:  {cur_stock}\nFinancials:   {cur_fin}\nIndustry:     {cur_ind}\nFund Flow:    {cur_fund}\nWeb Search:   {cur_ws_provider} ({'已配置' if cur_ws_key else '未配置'})", language=None)

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
