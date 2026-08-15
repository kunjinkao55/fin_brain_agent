# -*- coding: utf-8 -*-
"""Layer B — analyst 推理质量真实评估 CLI（需要 LLM）。

用冻结的黄金 collected_data 跑真实 analyst 节点（调 LLM），度量：
数字幻觉率 / 字段完整率 / 黄金事实命中率 / 结构化通过率。

用途（开发期调试定位）：
    - 改了 ANALYST_PROMPT / RAG 注入 → 跑本脚本看推理质量变化
    - 幻觉率上升 → 检查数据注入与 grounding 是否回归
    - 黄金事实未命中 → 检查 prompt 是否引导 LLM 引用正确事实

用法:
    python tests/run_analyst_eval.py [--runs N] [--symbol 301583]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台 GBK 下 UTF-8 源码中文会乱码，强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    runs = 1
    symbol = "301583"
    for i, a in enumerate(sys.argv):
        if a == "--runs" and i + 1 < len(sys.argv):
            runs = int(sys.argv[i + 1])
        elif a == "--symbol" and i + 1 < len(sys.argv):
            symbol = sys.argv[i + 1]

    from backend.eval_suite import evaluate_analyst, format_analyst_report

    print(f"评估标的: {symbol}，运行 {runs} 次（真实 LLM）...")
    report = evaluate_analyst(symbol=symbol, runs=runs)
    print(format_analyst_report(report))
    print("=" * 64)
    if report.get("平均幻觉率") is not None:
        print(f"阈值检查: 幻觉率 {report['平均幻觉率']:.1%} / "
              f"黄金事实命中 {report['平均黄金事实命中率']:.0%} / "
              f"字段完整 {report['平均字段完整率']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
