"""
FinBrain Agent Evaluation — 量化评估 Agent 输出的稳定性、完整性和可靠性。
不评估投资建议是否正确（那是回测的事），只评估 Agent 工程质量。
"""

import time, json, re
from collections import defaultdict


def evaluate_stock(symbol: str, runs: int = 10) -> dict:
    """对单只股票运行 N 次分析，收集评分一致性、字段完整率、工具成功率。"""
    from backend.agent import build_graph

    scores = []
    field_rates = []
    tool_rates = []
    errors = 0
    latencies = []

    _REQUIRED_FIELDS = [
        "投资逻辑", "评分", "亮点", "风险", "业绩驱动力", "关键信号",
        "估值水位", "情景估值", "催化剂", "证伪条件", "操作建议", "综合结论"
    ]

    graph = build_graph()
    for i in range(runs):
        tid = f"eval_{symbol}_{i}_{int(time.time()*1000)}"
        try:
            t0 = time.time()
            r = graph.invoke(
                {"messages": [], "user_question": f"分析{symbol}",
                 "collected_data": "", "analysis": "", "report": "", "processing_log": []},
                {"configurable": {"thread_id": tid}}
            )
            latencies.append((time.time() - t0) * 1000)

            analysis = r.get("analysis", "")
            report = r.get("report", "")
            score = _extract_score(report) or _extract_score(analysis)
            if score is not None:
                scores.append(score)

            # 字段完整率
            present = sum(1 for f in _REQUIRED_FIELDS if f in analysis)
            field_rates.append(present / len(_REQUIRED_FIELDS) * 100)

            # 工具成功率
            pl = r.get("processing_log", [])
            for entry in pl:
                if entry.get("phase") == "Data":
                    actions = entry.get("actions", [])
                    if actions:
                        ok = sum(1 for a in actions if a.get("status") == "✅")
                        tool_rates.append(ok / len(actions) * 100)
                    break
        except Exception as e:
            errors += 1

    # 聚合指标
    return {
        "代码": symbol,
        "运行次数": runs,
        "错误次数": errors,
        "评分一致性": _consistency(scores),
        "评分均值": round(sum(scores)/len(scores), 1) if scores else 0,
        "评分标准差": round(_stddev(scores), 2) if len(scores) > 1 else 0,
        "字段完整率": round(sum(field_rates)/len(field_rates), 1) if field_rates else 0,
        "工具成功率": round(sum(tool_rates)/len(tool_rates), 1) if tool_rates else 0,
        "平均延迟_ms": round(sum(latencies)/len(latencies)) if latencies else 0,
        "原始分数": scores,
    }


def _extract_score(text: str) -> float | None:
    """从分析文本/报告中提取加权总分。"""
    # 尝试多种格式
    for pat in [r'加权总分[:\s]*([\d.]+)/100', r'加权总分[:\s]*([\d.]+)',
                r'"加权总分":\s*([\d.]+)']:
        m = re.search(pat, text)
        if m: return float(m.group(1))
    return None


def _consistency(scores: list) -> float:
    """评分一致性：变异系数越小越一致。100%=完全相同。"""
    if len(scores) < 2: return 100.0
    std = _stddev(scores)
    mean = sum(scores) / len(scores)
    if mean == 0: return 100.0
    cv = std / mean
    return round(max(0, 100 - cv * 100), 1)


def _stddev(values: list) -> float:
    if len(values) < 2: return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
