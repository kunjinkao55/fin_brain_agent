# -*- coding: utf-8 -*-
"""三层评估汇总 CLI — 一键跑全部离线 CI + 可选真实诊断。

用法:
    python tests/run_eval_all.py                    # 三层 CI 门槛（离线，推荐）
    python tests/run_eval_all.py --real             # CI + 真实 LLM 推理评估
    python tests/run_eval_all.py --real --data      # CI + 真实 LLM + 真实数据抓取
    python tests/run_eval_all.py --symbols 301583,600131

输出: 逐层汇总 + 失败摘要。任何离线断言失败返回非零退出码（供 CI 用）。
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台 GBK 下 UTF-8 源码中文会乱码，强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _run(py_args: list[str]) -> tuple[int, str]:
    """运行一个子进程，返回 (退出码, 输出)。"""
    cmd = [sys.executable, "-m", "pytest"] if False else [sys.executable] + py_args
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _run_test_script(script: str, extra: list[str] | None = None) -> tuple[int, str]:
    """运行一个 tests/ 下的 Python 脚本，额外参数透传。"""
    return _run([os.path.join(_TESTS_DIR, script)] + (extra or []))


def _section(title: str, code: int, tail: str = "ok") -> None:
    mark = "✅" if code == 0 else "❌"
    print(f"{mark} {title}  ({tail})")
    if code != 0:
        print(f"   退出码 {code}")


def main() -> int:
    argv = sys.argv[1:]
    want_real = "--real" in argv
    want_data = "--data" in argv
    symbols = None
    if "--symbols" in argv:
        i = argv.index("--symbols")
        if i + 1 < len(argv):
            symbols = argv[i + 1]

    print("=" * 64)
    print("FinBrain 评估体系 — 三层汇总")
    print("=" * 64)

    failures = []

    # ---- Layer A: 数据管道（离线 CI）----
    code, out = _run_test_script("test_data_eval.py")
    _section("Layer A  数据管道黄金比对 (11 项 CI)", code,
             "FAIL" if code else "ok")
    if code:
        failures.append("test_data_eval")
        print(out[-2000:])

    # ---- Layer B: analyst 推理（离线 CI）----
    code, out = _run_test_script("test_analyst_eval.py")
    _section("Layer B  analyst 推理质量 (17 项 CI)", code,
             "FAIL" if code else "ok")
    if code:
        failures.append("test_analyst_eval")
        print(out[-2000:])

    # ---- Layer C: 审查链 fault-injection（离线 CI）----
    code, out = _run_test_script("test_fault_injection.py")
    _section("Layer C  审查链 fault-injection (4 项 CI)", code,
             "FAIL" if code else "ok")
    if code:
        failures.append("test_fault_injection")
        print(out[-2000:])

    # ---- 真实诊断（可选）----
    if want_real:
        print("\n[真实 LLM] analyst 推理质量...")
        code, out = _run_test_script("run_analyst_eval.py", ["--runs", "1"])
        print(out[-1500:])
        if code:
            failures.append("run_analyst_eval(real)")

    if want_data:
        syms = symbols or "301583"
        print("\n[真实数据] 数据管道黄金比对...")
        code, out = _run_test_script("run_data_eval.py", ["--symbols", syms])
        print(out[-1500:])
        if code:
            failures.append("run_data_eval(real)")

    print("=" * 64)
    if failures:
        print(f"评估失败: {', '.join(failures)}")
        return 1
    print("三层离线评估全部通过 ✅（真实诊断若开启请人工核对上方输出）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
