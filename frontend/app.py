"""
FinBrain Streamlit 前端 — 暗色主题
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import json

st.set_page_config(page_title="FinBrain", layout="wide")

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
                  font-family: Consolas,monospace; font-size: 13px; line-height: 1.7;
                  white-space: pre-wrap; border-radius: 4px; }
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
    page = st.radio("", ["Chat", "Portfolio", "Analysis", "Settings"], label_visibility="collapsed")
    st.divider()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat"
    st.button("Clear Chat", on_click=lambda: st.session_state.chat_history.clear())
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

def run_agent(user_input: str) -> str:
    agents = get_agents()
    req_type = agents["classify"](user_input)
    msgs = _to_lc(st.session_state.chat_history) + [{"role": "user", "content": user_input}]
    if req_type == "phantom":
        st.session_state.mode = "Phantom Hunter"
        return agents["phantom"].invoke({"messages": msgs})["messages"][-1].content
    elif req_type == "analysis":
        st.session_state.mode = "Deep Analysis"
        r = agents["graph"].invoke({"messages": msgs, "user_question": user_input,
                                     "collected_data":"", "analysis":"", "report":""})
        return r.get("report") or r["messages"][-1].content
    else:
        st.session_state.mode = "Chat"
        return agents["chat"].invoke({"messages": msgs})["messages"][-1].content


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
                    reply = run_agent(prompt)
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
                    reply = run_agent(f"分析{sym}的财报和估值")
                    st.markdown(f'<div class="report-block">{reply}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check that API Key is set in Settings page and restart the app.")


# ========== Settings ==========
elif page == "Settings":
    st.header("Settings")
    tab1,tab2,tab3 = st.tabs(["LLM Provider","Tools","System"])
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
