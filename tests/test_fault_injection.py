# -*- coding: utf-8 -*-
"""审查链 fault-injection 回归测试 — 确定性守卫诊断。

对每条注入故障断言：
    1. 阳性检测：注入错误必须被对应守卫检出（100% 检出基线）
    2. 阴性对照：干净黄金数据必须零误报（防守卫过度敏感）

全部基于黄金数据离线运行，不访问网络与 LLM。
运行: python tests/test_fault_injection.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")


class TestFaultInjection(unittest.TestCase):
    """注入已知错误类型 → 对应守卫必须检出。"""

    def setUp(self):
        from backend.eval_suite.faults import build_faults
        self.faults = build_faults()

    def test_faults_defined(self):
        """故障目录非空，id 唯一。"""
        self.assertTrue(len(self.faults) >= 10, f"故障数量不足: {len(self.faults)}")
        ids = [f.id for f in self.faults]
        self.assertEqual(len(ids), len(set(ids)), "fault id 重复")

    def test_all_faults_detected(self):
        """阳性基线：每条注入故障都必须被检出。"""
        for f in self.faults:
            with self.subTest(fault=f.id):
                violations = f.detect(f.faulted)
                self.assertTrue(len(violations) > 0,
                                f"[{f.id}] {f.name} 注入后未被检出（期望守卫: {f.expected_guard}）")

    def test_no_false_positive_on_golden(self):
        """阴性对照：干净黄金数据必须零误报。"""
        for f in self.faults:
            with self.subTest(fault=f.id):
                violations = f.detect(f.golden)
                self.assertEqual(len(violations), 0,
                                 f"[{f.id}] {f.name} 干净数据误报: {violations}")


class TestFaultInjectionReport(unittest.TestCase):
    """诊断报告聚合正确。"""

    def test_report_structure(self):
        from backend.eval_suite.harness import run_fault_injection
        report = run_fault_injection()
        self.assertEqual(report["检出率"], 1.0, "黄金基线必须 100% 检出")
        self.assertEqual(report["阴性误报"], 0, "黄金基线必须零误报")
        self.assertEqual(report["检出数"], report["总故障数"])
        # 分守卫覆盖非空
        guard_keys = list(report["分守卫"].keys())
        self.assertIn("consistency.filter_untraceable_issues", guard_keys)
        self.assertIn("calc_engine.verify_report_text", guard_keys)
        self.assertTrue(any(k.startswith("consistency.check_invariants") for k in guard_keys),
                        f"缺少 check_invariants 守卫: {guard_keys}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
