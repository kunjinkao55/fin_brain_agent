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
    page = st.radio("", ["Market", "Chat", "Portfolio", "Analysis", "Settings"], label_visibility="collapsed")
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

def run_agent(user_input: str) -> tuple[str, list]:
    """返回 (回复文本, 工具调用记录列表)"""
    agents = get_agents()
    req_type = agents["classify"](user_input)
    msgs = _to_lc(st.session_state.chat_history) + [{"role": "user", "content": user_input}]
    tracker = ToolCallTracker()
    cfg = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "callbacks": [tracker],
    }

    if req_type == "phantom":
        st.session_state.mode = "Phantom Hunter"
        reply = agents["phantom"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content
    elif req_type == "analysis":
        st.session_state.mode = "Deep Analysis"
        r = agents["graph"].invoke({"messages": msgs, "user_question": user_input,
                                     "collected_data":"", "analysis":"", "report":"",
                                     "processing_log": []}, config=cfg)
        reply = r.get("report") or r["messages"][-1].content
        # 提取流水线日志
        proc_log = r.get("processing_log", [])
        if proc_log:
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
        st.session_state.mode = "Chat"
        reply = agents["chat"].invoke({"messages": msgs}, config=cfg)["messages"][-1].content

    return reply, tracker.records


# ========== Market ==========
if page == "Market":
    st.header("Market Monitor")

    # ---- 刷新 ----
    if st.button("Refresh Data", type="primary"):
        st.rerun()

    # ---- 涨跌全景 ----
    from backend.tools import get_market_breadth
    breadth = get_market_breadth()
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

    # ---- 全板块资金流对比图（主位，所有板块在同一坐标系） ----
    from backend.tools import get_sector_fund_flow
    sector_data = get_sector_fund_flow(100)  # 全部板块
    all_sectors = sector_data.get("列表", [])

    if all_sectors:
        names = [s["板块"] for s in all_sectors]
        nets = [s["净额(亿)"] for s in all_sectors]
        abs_nets = [abs(n) for n in nets]  # 绝对值
        colors = ["#cc3333" if n >= 0 else "#2e7d32" for n in nets]

        fig = go.Figure(data=[go.Bar(
            x=abs_nets, y=names, orientation='h',
            marker_color=colors,
            text=[f"{n:+.2f}亿" for n in nets],
            textposition='outside',
            textfont=dict(color='#ddd', size=9),
        )])
        fig.update_layout(
            title=f"Sector Fund Flow ({len(all_sectors)} sectors)",
            height=max(600, len(names) * 20),
            margin=dict(l=10, r=60, t=40, b=10),
            paper_bgcolor="#111", plot_bgcolor="#111",
            font=dict(color="#ddd"),
            xaxis=dict(title="|Net Flow| (亿)  Red=Inflow  Green=Outflow", showgrid=True, gridcolor="#333"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 下方表格
        with st.expander(f"Sector Fund Flow Details ({len(all_sectors)} sectors)", expanded=False):
            lines = [f"{'Sector':<12} {'Chg':>8} {'In(亿)':>10} {'Out(亿)':>10} {'Net(亿)':>10}"]
            lines.append("-" * 55)
            for s in all_sectors:
                net = s["净额(亿)"]
                color = "#cc3333" if net >= 0 else "#2e7d32"
                lines.append(
                    f"<span style='color:{color}'>{s['板块']:<12} {s['涨跌幅']:>8} "
                    f"{s['流入(亿)']:>10.2f} {s['流出(亿)']:>10.2f} {net:>+10.2f}</span>"
                )
            st.markdown("<pre style='font-size:12px'>" + "\n".join(lines) + "</pre>",
                        unsafe_allow_html=True)
    else:
        st.warning("No sector data available for this date")


# ========== Chat ==========
if page == "Chat":
    st.header("AI Chat")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])
    if prompt := st.chat_input("Ask FinBrain..."):
        with st.chat_message("user"): st.text(prompt)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    reply, tool_logs = run_agent(prompt)
                    # 处理流水线展示
                    pipeline = tool_logs or []
                    if pipeline:
                        with st.expander(f"Pipeline: Data({len(pipeline)} tools) -> Analysis -> Report", expanded=False):
                            st.caption("Phase 1: Data Collection")
                            for log in pipeline:
                                icon = {"running":"...","done":"OK","error":"ERR"}.get(log["status"],"?")
                                st.caption(f"  [{icon}] {log['tool']}({log['input'][:60]})")
                            st.caption(f"Phase 2: Analysis & Scoring")
                            st.caption(f"Phase 3: Report Formatting (output {len(reply)} chars)")
                    st.text(reply)
                except Exception as e:
                    reply = f"[Error] {e}"; st.text(reply)

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})


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

    sym = st.text_input("Stock Code", placeholder="300502")
    btn = st.button("Analyze", type="primary")

    if btn:
        if not sym:
            st.warning("Please enter a stock code")
        else:
            with st.spinner(f"Analyzing {sym}... (data collection may take 20-40s)"):
                try:
                    reply, _ = run_agent(f"分析{sym}的财报和估值")
                    st.html(f'<div class="report-block"><pre>{reply}</pre></div>')
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check that API Key is set in Settings page and restart the app.")


# ========== Settings ==========
elif page == "Settings":
    st.header("Settings")
    tab0,tab1,tab2,tab3 = st.tabs(["Strategy","LLM Provider","Tools","System"])
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
        cur_key = os.getenv("DEEPSEEK_API_KEY","") or os.getenv("OPENAI_API_KEY","") or os.getenv("ANTHROPIC_API_KEY","")

        provider = st.selectbox("Provider", ["deepseek","openai","anthropic"],
                                index=["deepseek","openai","anthropic"].index(cur_provider) if cur_provider in ["deepseek","openai","anthropic"] else 0)
        api_key = st.text_input("API Key", type="password", value=cur_key, placeholder="sk-...", key="settings_apikey")
        base_url = st.text_input("Base URL (optional)", value="https://api.deepseek.com" if provider=="deepseek" else "", key="settings_url")
        model = st.text_input("Model", value="deepseek-chat" if provider=="deepseek" else "gpt-4o", key="settings_model")

        if st.button("Apply & Save", type="primary"):
            os.environ["LLM_PROVIDER"] = provider
            if api_key:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
            if base_url:
                os.environ["LLM_BASE_URL"] = base_url
            # 写入 configs/.env 持久化
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", ".env")
            with open(env_path, "w") as f:
                f.write(f"LLM_PROVIDER={provider}\n")
                if api_key:
                    f.write(f"{provider.upper()}_API_KEY={api_key}\n")
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
        c1,c2 = st.columns(2)
        with c1: st.number_input("Compress Trigger", value=int(os.getenv("COMPRESS_TRIGGER","12")), min_value=4, key="ct")
        with c2: st.number_input("Compress Keep", value=int(os.getenv("COMPRESS_KEEP","6")), min_value=2, key="ck")
        if st.button("Save Compress Config", type="primary"):
            os.environ["COMPRESS_TRIGGER"] = str(st.session_state.get("ct",12))
            os.environ["COMPRESS_KEEP"] = str(st.session_state.get("ck",6))
            st.success("Applied")
    st.divider(); st.caption("FinBrain v0.2")
