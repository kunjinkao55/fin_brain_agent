"""
FinBrain 客户端 — 用户本地 LLM 调用封装。
从本地 .env 读取 Key，调用 LLM API。Key 永不传到服务器。
"""

import os, json, logging

logger = logging.getLogger(__name__)

_LLM = None


def _get_local_llm():
    """获取本地 LLM 实例（从客户端 .env 读取 Key）"""
    global _LLM
    if _LLM is not None:
        return _LLM

    provider = os.getenv("LLM_PROVIDER", "deepseek")
    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        _LLM = ChatOpenAI(
            model="deepseek-chat", temperature=0, max_tokens=4096,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        _LLM = ChatOpenAI(
            model="gpt-4o", temperature=0, max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        from langchain_anthropic import ChatAnthropic
        _LLM = ChatAnthropic(
            model="claude-sonnet-5", temperature=0, max_tokens=4096,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    return _LLM


def invoke_remote_analysis(api_base: str, question: str, symbols: list[str],
                           stream: bool = False) -> str:
    """远程模式完整流程：调 API 拿 Prompt → 本地 LLM 推理 → 返回报告文本。
    设置 stream=True 可在终端流式输出。"""
    import urllib.request

    # 1) 从服务器获取 Prompt + 数据
    payload = json.dumps({"question": question, "symbols": symbols}).encode("utf-8")
    url = f"{api_base.rstrip('/')}/api/prompt/analysis"
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "User-Agent": "FinBrain-Client/2.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"[Error] Failed to reach Data API at {url}: {e}"

    if "error" in prompt_data:
        return f"[Error] {prompt_data['error']}"

    # 2) 本地 LLM 推理
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _get_local_llm()
    response = llm.invoke([
        SystemMessage(content=prompt_data["system_prompt"]),
        HumanMessage(content=prompt_data["user_prompt"]),
    ])

    # 3) 本地 Reporter 后处理（评分覆盖 + 估值水位 + 现金流预警）
    raw_analysis = response.content
    return _local_reporter_postprocess(raw_analysis, prompt_data.get("data", []))


def _local_reporter_postprocess(raw_analysis: str, data: list[dict]) -> str:
    """本地执行 Reporter 后处理：评分覆盖 + 估值水位 + 现金流预警 + 格式化"""
    import re
    from backend.tools import format_report, _format_compare_section, calculate_scores
    from backend.scoring import compute_investment_rating

    # JSON 解析
    raw_stripped = raw_analysis.strip()
    analysis_data = None
    try:
        analysis_data = json.loads(raw_stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw_stripped):
            if ch in '[{':
                try:
                    analysis_data, _ = decoder.raw_decode(raw_stripped[i:])
                    break
                except json.JSONDecodeError:
                    continue

    if analysis_data is None:
        return raw_analysis  # 解析失败，返回原始输出

    # 对每只股票后处理
    items = analysis_data if isinstance(analysis_data, list) else [analysis_data]

    for item in items:
        if not isinstance(item, dict):
            continue
        sym = item.get("代码", "")
        # 从预取数据中找匹配的股票
        stock_data = next((d for d in data if d.get("代码") == sym), {})
        pre_scores = stock_data.get("预计算分数", {})
        val_data = stock_data.get("估值", [])
        price_info = stock_data.get("行情", {})
        ind_info = stock_data.get("行业", {})
        fin_info = stock_data.get("财报", {})

        # 评分覆盖
        for dim in ["盈利能力", "成长性", "财务健康", "估值合理"]:
            if dim in pre_scores:
                item.setdefault("评分", {})[dim] = pre_scores[dim]

        # 估值水位
        latest_val = val_data[0] if val_data else {}
        stock_price = float(price_info.get("price", 0) or 0) if isinstance(price_info, dict) else 0
        eps = float(latest_val.get("每股收益", 0) or 0)
        bps = float(latest_val.get("每股净资产", 0) or 0)
        shares = float(latest_val.get("总股本", 0) or 0)
        pe_val = stock_price / eps if eps > 0 else 0
        pb_val = stock_price / bps if bps > 0 else 0
        mktcap = shares * stock_price / 1e8 if shares > 0 else 0
        item["估值水位"] = {
            "PE": f"{pe_val:.0f}", "PB": f"{pb_val:.1f}",
            "市值": f"{mktcap:.0f}亿" if mktcap > 0 else "-",
        }

        # Q1 现金流预警
        cf_list = fin_info.get("现金流", [])
        q1_cf = next((r for r in cf_list if r.get("报告期") == "一季报"), None)
        if q1_cf:
            cf_val = float(q1_cf.get("经营现金流净额", 0) or 0)
            if cf_val < 0:
                profit_list = fin_info.get("利润表", [])
                q1_p = next((r for r in profit_list if r.get("报告期") == "一季报"), {})
                q1_profit = float(q1_p.get("归母净利润", 0) or 0)
                warning = (f"Q1经营现金流{cf_val/1e8:.1f}亿(净流出), "
                           f"与净利润{q1_profit/1e8:.1f}亿背离。需关注Q2是否改善。")
                item["风险"] = (item.get("风险", []) if isinstance(item.get("风险"), list) else []) + [warning]

        # 投资决策
        try:
            industry = ind_info.get("行业", ind_info.get("industry_name", "")) if isinstance(ind_info, dict) else ""
            roe = float(latest_val.get("ROE(%)", 0) or 0)
            debt = float(latest_val.get("资产负债率(%)", 50) or 50)
            profile = item.get("公司画像", {}) if isinstance(item, dict) else {}
            ctype = profile.get("公司类型", "价值型") if isinstance(profile, dict) else "价值型"
            llm_scores = item.get("评分", {})
            decision = compute_investment_rating(
                company_type=ctype,
                financial_scores={k: pre_scores.get(k, {}) for k in ["盈利能力","成长性","财务健康","估值合理"]},
                llm_scores={"行业前景": llm_scores.get("行业前景",{}), "资金认可": llm_scores.get("资金认可",{})},
                eps=eps, stock_price=stock_price, industry=industry, roe=roe, debt=debt,
            )
            item["投资评级"] = decision
        except Exception:
            pass

    # 格式化输出
    compare_text = ""
    cleaned = []
    for item in items:
        if isinstance(item, dict) and "对比分析" in item:
            compare_text = _format_compare_section(item.pop("对比分析"))
        cleaned.append(item)
    score_cards = [format_report(it) for it in cleaned if isinstance(it, dict)]
    score_text = "\n\n".join(score_cards)
    if compare_text:
        score_text += "\n\n" + compare_text

    return score_text


def invoke_remote_phantom(api_base: str, question: str = "查找近期潜力妖股") -> str:
    """远程妖股模式：拿快照 + 本地 LLM 推演"""
    import urllib.request
    payload = json.dumps({"question": question}).encode("utf-8")
    url = f"{api_base.rstrip('/')}/api/prompt/phantom"
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json", "User-Agent": "FinBrain-Client/2.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"[Error] Phantom API unreachable: {e}"

    # 本地 LLM 推演
    from langchain_core.messages import SystemMessage, HumanMessage
    llm = _get_local_llm()
    snapshot_text = json.dumps(prompt_data.get("snapshot", {}), ensure_ascii=False, indent=2)
    user_prompt = f"{question}\n\n今日市场数据快照:\n{snapshot_text}"
    response = llm.invoke([
        SystemMessage(content=prompt_data.get("system_prompt", "")),
        HumanMessage(content=user_prompt),
    ])
    return response.content
