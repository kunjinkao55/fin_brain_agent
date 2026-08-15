"""
诊断运行器 — 对黄金数据跑全部 fault-injection，聚合检出率报告。

每次输出：
    1. 逐条故障：是否检出 + 检出证据 + 期望守卫
    2. 分守卫统计：哪个守卫拦住了哪些故障（回归定位）
    3. 分阶段统计：critics / repair / report / audit 检出率

使用场景（开发期调试定位）：
    - 改了某条守卫代码 → 跑 run_fault_injection()，看哪些故障不再被检出
    - 报告数字错乱 → 看 report 阶段检出率是否下滑
    - 新增守卫 → 在 faults.py 注册新故障并断言 100% 检出
"""

from dataclasses import dataclass, field


@dataclass
class FaultResult:
    id: str
    category: str
    name: str
    expected_guard: str
    golden_clean: bool      # 干净数据零误报
    faulted_caught: bool    # 注入错误被检出
    evidence: list = field(default_factory=list)  # 检出证据（违规描述）
    error: str = ""


def run_fault_injection() -> dict:
    """对全部故障跑检测器，返回聚合报告。"""
    from backend.eval_suite.faults import build_faults

    results: list[FaultResult] = []
    for f in build_faults():
        res = FaultResult(
            id=f.id, category=f.category, name=f.name,
            expected_guard=f.expected_guard,
            golden_clean=False, faulted_caught=False,
        )
        # 阴性对照：干净数据必须零误报
        try:
            clean_violations = f.detect(f.golden)
            res.golden_clean = len(clean_violations) == 0
            if clean_violations and not res.error:
                res.error = "阴性对照误报: " + " | ".join(str(v) for v in clean_violations[:3])
        except Exception as e:
            res.golden_clean = False
            res.error = f"阴性对照异常: {e}"
        # 阳性检测：注入错误必须被检出
        try:
            violations = f.detect(f.faulted)
            res.faulted_caught = len(violations) > 0
            res.evidence = [str(v) for v in violations[:3]]
        except Exception as e:
            res.faulted_caught = False
            res.error = (res.error + " | " if res.error else "") + f"阳性检测异常: {e}"
        results.append(res)

    return _aggregate(results)


def _aggregate(results: list[FaultResult]) -> dict:
    total = len(results)
    caught = sum(1 for r in results if r.faulted_caught)
    clean = sum(1 for r in results if r.golden_clean)

    by_guard: dict[str, dict] = {}
    by_category: dict[str, dict] = {}
    for r in results:
        g = by_guard.setdefault(r.expected_guard, {"caught": 0, "total": 0, "faults": []})
        g["total"] += 1
        if r.faulted_caught:
            g["caught"] += 1
        g["faults"].append(r.id)

        c = by_category.setdefault(r.category, {"caught": 0, "total": 0, "ids": []})
        c["total"] += 1
        if r.faulted_caught:
            c["caught"] += 1
        c["ids"].append(r.id)

    return {
        "总故障数": total,
        "检出数": caught,
        "检出率": round(caught / total, 3) if total else 0.0,
        "阴性零误报数": clean,
        "阴性误报": total - clean if total else 0,
        "分守卫": {
            k: {"检出": v["caught"], "总数": v["total"], "检出率": round(v["caught"] / v["total"], 2), "故障": v["faults"]}
            for k, v in sorted(by_guard.items())
        },
        "分阶段": {
            k: {"检出": v["caught"], "总数": v["total"], "检出率": round(v["caught"] / v["total"], 2), "故障": v["ids"]}
            for k, v in sorted(by_category.items())
        },
        "明细": [
            {
                "id": r.id, "阶段": r.category, "名称": r.name,
                "期望守卫": r.expected_guard,
                "检出": r.faulted_caught, "零误报": r.golden_clean,
                "证据": r.evidence, "异常": r.error,
            }
            for r in results
        ],
    }


def format_report(report: dict) -> str:
    """人类可读的报告文本。"""
    lines = []
    lines.append("=" * 64)
    lines.append("FinBrain 审查链 fault-injection 诊断报告")
    lines.append("=" * 64)
    lines.append(f"总检出率: {report['检出数']}/{report['总故障数']} = {report['检出率']:.0%}")
    lines.append(f"阴性零误报: {report['阴性零误报数']}/{report['总故障数']}（误报 {report['阴性误报']}）")

    lines.append("\n[分阶段]")
    for k, v in report["分阶段"].items():
        lines.append(f"  {k:<8} 检出 {v['检出']}/{v['总数']} ({v['检出率']:.0%})  {','.join(v['故障'])}")

    lines.append("\n[分守卫]")
    for k, v in report["分守卫"].items():
        mark = "OK" if v["检出率"] == 1.0 else "!!回归"
        lines.append(f"  {mark:<4} {v['检出']}/{v['总数']}  {k}")

    lines.append("\n[逐条明细]")
    for r in report["明细"]:
        status = "检出" if r["检出"] else "未检出 !!"
        fp = "零误报" if r["零误报"] else "误报!!"
        lines.append(f"  {r['id']} [{r['阶段']}] {r['名称']:<28} {status} / {fp}")
        lines.append(f"       期望守卫: {r['期望守卫']}")
        if r["证据"]:
            lines.append(f"       证据: {' | '.join(r['证据'])}")
        if r["异常"]:
            lines.append(f"       异常: {r['异常']}")

    return "\n".join(lines)
