"""
Layer B — 推理链组件级评估（analyst 节点）。

目的（开发期调试定位）：冻结数据层（黄金 collected_data，已知事实），
跑 analyst 节点，度量 LLM 推理输出质量：

    1. 数字幻觉率   — 输出中无出处的硬数字比例（复用 grounding 机制）
    2. 字段完整率   — AnalystOutput 关键字段是否齐全
    3. 黄金事实命中 — 已知事实（TTM EPS/PE/PB/发行价/现价）是否反映在输出
    4. 结构化质量   — JSON 可解析、评级合法、情景价格单调

设计：
    evaluate_analysis(text, collected)  纯函数指标 —— 可离线进 CI，不调 LLM
    evaluate_analyst(symbol, runs)      真实运行器 —— 调 analyst_node(LLM)，供 CLI 周度监控

黄金数据来自 golden_pool（默认 301583，可按 symbol 切换板块）。
与现有项目机制一致：幻觉判定复用 backend.consistency 的 grounding 池与容差，
不发明新校验。
"""

import json
import re

# 黄金股票池：build_golden_collected / build_golden_facts 按 symbol 取数
from backend.eval_suite.golden_pool import build_golden_collected, build_golden_facts

# 默认标的（托伦斯，与 tuolunsi golden 一致）
DEFAULT_SYMBOL = "301583"
GOLDEN_COLLECTED = build_golden_collected(DEFAULT_SYMBOL)
GOLDEN_FACTS = build_golden_facts(DEFAULT_SYMBOL)

# AnalystOutput 关键字段（缺失即视为推理不完整）
REQUIRED_FIELDS = [
    "代码", "名称", "投资逻辑链", "评分", "亮点", "风险", "业绩驱动力",
    "关键信号", "估值水位", "情景估值", "催化剂", "证伪条件",
    "操作建议", "投资评级", "综合结论",
]


# ============================================================
#  纯函数指标（不调 LLM，可离线 CI）
# ============================================================

def parse_analysis(text: str) -> dict | None:
    """解析分析 JSON。支持单对象 / 数组（取第一个）。失败返回 None。"""
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        try:
            data = json.loads(re.sub(r"```(?:json)?|```", "", text).strip())
        except (ValueError, TypeError):
            return None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def hallucination_metrics(text: str, collected: str) -> dict:
    """数字幻觉率：输出中带单位的硬数字里，无出处（grounding 池±容差）的比例。"""
    from backend.consistency import has_new_hard_numbers, extract_hard_numbers
    total = extract_hard_numbers(text)
    new = has_new_hard_numbers(collected, text)
    rate = round(len(new) / len(total), 3) if total else 0.0
    return {
        "总硬数字": len(total),
        "无出处硬数字": len(new),
        "幻觉率": rate,
        "无出处示例": [f"{v:.4g}{u}" for v, u in new[:5]],
    }


def field_completeness(text: str) -> dict:
    """字段完整率：AnalystOutput 关键字段的填充比例。"""
    data = parse_analysis(text) or {}
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    present = len(REQUIRED_FIELDS) - len(missing)
    return {
        "完整率": round(present / len(REQUIRED_FIELDS), 3) if REQUIRED_FIELDS else 1.0,
        "缺失字段": missing,
    }


def _any_number_in(text: str, target: float, tol: float) -> bool:
    """文本中是否存在与 target 偏差 ≤tol（相对）的数字（裸数字扫描）。"""
    from backend.consistency import _extract_bare_numbers
    for v in _extract_bare_numbers(text):
        if v == 0 and target == 0:
            return True
        denom = max(abs(v), abs(target))
        if denom > 0 and abs(v - target) / denom <= tol:
            return True
    return False


def golden_fact_hits(text: str, facts: list | None = None) -> dict:
    """黄金事实命中：每个已知事实是否反映在输出。facts 默认取 GOLDEN_FACTS。"""
    facts = facts if facts is not None else GOLDEN_FACTS
    hits, misses = [], []
    for name, value, tol in facts:
        if _any_number_in(text, value, tol):
            hits.append(name)
        else:
            misses.append(name)
    return {
        "命中": hits,
        "未命中": misses,
        "命中率": round(len(hits) / len(facts), 3) if facts else 1.0,
    }


def structural_checks(text: str) -> dict:
    """结构化质量：JSON 可解析、评级合法、情景价格单调。"""
    data = parse_analysis(text)
    if data is None:
        return {"json_ok": False, "rating_ok": False, "scenario_monotonic": False,
                "notes": ["分析 JSON 无法解析"]}
    notes = []
    # 评级合法
    rating = data.get("投资评级")
    if isinstance(rating, dict):
        level = str(rating.get("评级", "") or "").upper()
        rating_ok = level in ("BUY", "HOLD", "SELL", "AVOID", "买入", "持有", "卖出")
        if not rating_ok:
            notes.append(f"评级非法: {rating.get('评级')}")
    else:
        rating_ok = False
        notes.append("缺少 投资评级.评级")
    # 情景价格单调
    sc = data.get("情景估值")
    prices = []
    if isinstance(sc, dict):
        for k in ("悲观", "基准", "乐观"):
            s = sc.get(k)
            if isinstance(s, dict):
                try:
                    prices.append(float(str(s.get("价格", "")).replace("元", "")))
                except (ValueError, TypeError):
                    prices.append(None)
    if len(prices) == 3 and all(p is not None for p in prices):
        mono = prices[0] <= prices[1] <= prices[2]
        if not mono:
            notes.append(f"情景价格倒挂: {prices}")
    elif prices:
        mono = False
        notes.append("情景价格缺失/无法解析")
    else:
        mono = True
        notes.append("无情景估值数据（跳过）")
    return {"json_ok": True, "rating_ok": rating_ok,
            "scenario_monotonic": mono, "notes": notes}


def evaluate_analysis(text: str, collected: str | None = None,
                      symbol: str = DEFAULT_SYMBOL) -> dict:
    """对单份分析输出计算全部纯函数指标。

    collected 缺省时按 symbol 从黄金股票池构造；facts 同步取自该股票。
    """
    if collected is None:
        collected = build_golden_collected(symbol)
    facts = build_golden_facts(symbol)
    return {
        "字段完整率": field_completeness(text),
        "数字幻觉": hallucination_metrics(text, collected),
        "黄金事实": golden_fact_hits(text, facts),
        "结构化": structural_checks(text),
    }


# ============================================================
#  真实运行器（调 analyst_node + LLM，供 CLI / 周度监控）
# ============================================================

def evaluate_analyst(symbol: str = DEFAULT_SYMBOL, runs: int = 1) -> dict:
    """冻结黄金 collected_data，跑 analyst 节点 N 次，聚合指标。"""
    from backend.agent import analyst_node

    collected = build_golden_collected(symbol)
    per_run = []
    for i in range(runs):
        state = {
            "collected_data": collected,
            "user_question": f"分析{symbol}",
            "processing_log": [],
        }
        try:
            out = analyst_node(state)
            analysis = out.get("analysis", "")
            metrics = evaluate_analysis(analysis, collected=collected, symbol=symbol)
            metrics["fallback_used"] = bool(
                any(l.get("phase") == "Analysis" and l.get("fallback_used")
                    for l in out.get("processing_log", [])))
            metrics["output_chars"] = len(analysis)
            metrics["error"] = None
        except Exception as e:
            metrics = {"error": str(e)}
        per_run.append(metrics)

    return _aggregate(per_run)


def _aggregate(per_run: list[dict]) -> dict:
    from statistics import mean
    ok_runs = [r for r in per_run if not r.get("error")]
    report = {
        "运行次数": len(per_run),
        "成功次数": len(ok_runs),
        "失败次数": len(per_run) - len(ok_runs),
        "错误": [r["error"] for r in per_run if r.get("error")],
    }
    if ok_runs:
        report["平均字段完整率"] = round(mean(r["字段完整率"]["完整率"] for r in ok_runs), 3)
        report["平均幻觉率"] = round(mean(r["数字幻觉"]["幻觉率"] for r in ok_runs), 3)
        report["平均黄金事实命中率"] = round(mean(r["黄金事实"]["命中率"] for r in ok_runs), 3)
        report["结构化通过率"] = round(mean(
            1.0 if (r["结构化"]["json_ok"] and r["结构化"]["rating_ok"]
                    and r["结构化"]["scenario_monotonic"]) else 0.0 for r in ok_runs), 3)
        report["fallback_使用次数"] = sum(1 for r in ok_runs if r.get("fallback_used"))
        report["逐次明细"] = per_run
    return report


def format_analyst_report(report: dict) -> str:
    """人类可读报告。"""
    lines = []
    lines.append("=" * 64)
    lines.append("FinBrain Layer B — analyst 推理质量评估报告")
    lines.append("=" * 64)
    lines.append(f"运行 {report['运行次数']} 次，成功 {report['成功次数']}，失败 {report['失败次数']}")
    if report.get("错误"):
        lines.append(f"错误: {report['错误'][:3]}")
    if "平均字段完整率" in report:
        lines.append(f"字段完整率     : {report['平均字段完整率']:.0%}")
        lines.append(f"数字幻觉率     : {report['平均幻觉率']:.1%}")
        lines.append(f"黄金事实命中率 : {report['平均黄金事实命中率']:.0%}")
        lines.append(f"结构化通过率   : {report['结构化通过率']:.0%}")
        if report.get("fallback_使用次数"):
            lines.append(f"结构化回退次数 : {report['fallback_使用次数']}")
    return "\n".join(lines)
