# -*- coding: utf-8 -*-
"""审查链 fault-injection 诊断 CLI — 打印检出率报告。

用法:
    python tests/run_fault_injection.py [--json]

用途（开发期调试定位）:
    1. 改了守卫代码 → 跑本脚本，看哪些故障不再被检出（回归定位）
    2. 报告数字错乱 → 看 report 阶段检出率
    3. 看每条故障的证据（被哪条守卫拦截、证据文本）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

# Windows 控制台 GBK 下 UTF-8 源码中文会乱码，强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> int:
    from backend.eval_suite import run_fault_injection, format_report

    report = run_fault_injection()
    if "--json" in sys.argv:
        import json
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    # 非零退出码：检出率<100% 或存在误报 → 便于 CI 中断
    if report["检出率"] < 1.0 or report["阴性误报"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
