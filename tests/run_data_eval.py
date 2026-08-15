# -*- coding: utf-8 -*-
"""Layer A — 数据管道黄金比对 CLI（需要网络）。

调用与 data_collector 相同的工具抓取真实数据，与黄金事实（上市公告书口径）比对，
定位数据层错误（股本未除权/财报滞后/字段缺失）—— 与 agent 推理错误分开。

用法:
    python tests/run_data_eval.py [--symbols 301583,600131]
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
    symbols = ["301583"]
    for i, a in enumerate(sys.argv):
        if a == "--symbols" and i + 1 < len(sys.argv):
            symbols = [s.strip() for s in sys.argv[i + 1].split(",") if s.strip()]

    from backend.eval_suite import evaluate_data_pipeline, format_pipeline_report

    print(f"评估标的: {symbols}（真实数据抓取）...")
    report = evaluate_data_pipeline(symbols)
    print(format_pipeline_report(report))
    print("=" * 64)
    print(f"通过率: {report['通过率']:.0%}")
    return 0 if report["通过率"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
