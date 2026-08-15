# -*- coding: utf-8 -*-
"""Layer A — 数据管道黄金比对的纯函数测试。

只测 extract_metrics / check_against_golden（不调 LLM、不访问网络）：
    1. 黄金数据（托伦斯口径）→ 提取正确 + 零违规
    2. 注入数据错误（股本未除权 / 归母错 / 报告期滞后 / 字段缺失）→ 违规检出

运行: python tests/test_data_eval.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

from backend.eval_suite.data import extract_metrics, check_against_golden


# 与黄金事实一致（托伦斯口径）的抓取结构
def _good_collected():
    return {
        "代码": "301583",
        "行情": {"当前价格": 174.0},
        "估值": {"data": [
            {"报告期": "2026-06-30 [半年报快报]", "总股本": 1.85473692e8,
             "每股净资产": 9.87, "每股收益": 0.53, "ROE(%)": 11.78},
        ]},
        "财报": {"利润表": [
            {"报告期": "一季报", "归母净利润": 0.1464e8, "扣非净利润": None},
            {"报告期": "年报", "归母净利润": 0.9818e8, "扣非净利润": None},
            {"报告期": "一季报", "归母净利润": 0.1600e8, "扣非净利润": None},
        ]},
        "公告": {"列表": [
            {"标题": "2026年半年度业绩快报", "日期": "2026-07-17",
             "快报数据": {"报告期": "2026-06-30 [半年报快报]"}},
        ]},
    }


def _mutate(collected, **overrides):
    """浅拷贝并按路径覆盖字段（测试注入用）。"""
    import copy
    c = copy.deepcopy(collected)
    if "估值" in overrides:
        c["估值"]["data"][0].update(overrides["估值"])
    if "利润表" in overrides:
        c["财报"]["利润表"] = overrides["利润表"]
    if "代码" in overrides:
        c["代码"] = overrides["代码"]
    return c


class TestExtractMetrics(unittest.TestCase):
    def test_golden_extracts_all(self):
        m = extract_metrics(_good_collected())
        self.assertAlmostEqual(m["总股本(股)"], 1.85473692e8, delta=1e2)
        self.assertAlmostEqual(m["每股净资产(元)"], 9.87, delta=0.01)
        self.assertAlmostEqual(m["2025年报归母(元)"], 0.9818e8, delta=1e5)
        self.assertAlmostEqual(m["2026Q1归母(元)"], 0.1464e8, delta=1e5)
        # TTM = 0.9818 + 0.1464 - 0.1600 = 0.9682 亿
        self.assertAlmostEqual(m["TTM归母(元)"], 0.9682e8, delta=1e5)
        self.assertAlmostEqual(m["TTM EPS"], 0.522, delta=0.002)
        self.assertEqual(m["最新报告期"], "一季报")
        self.assertTrue(m["含快报"])

    def test_missing_valuation(self):
        c = _good_collected()
        c["估值"] = {"data": []}
        m = extract_metrics(c)
        self.assertIsNone(m["总股本(股)"])
        self.assertIsNone(m["TTM EPS"])

    def test_incomplete_profit_no_ttm(self):
        c = _mutate(_good_collected(), 利润表=[
            {"报告期": "一季报", "归母净利润": 0.1464e8},
        ])
        m = extract_metrics(c)
        self.assertIsNone(m["TTM归母(元)"])


class TestCheckAgainstGolden(unittest.TestCase):
    def test_golden_passes(self):
        m = extract_metrics(_good_collected())
        self.assertEqual(check_against_golden(m, "301583"), [])

    def test_stale_share_capital_flagged(self):
        """股本未除权（发行前 1.39亿 vs 发行后 1.85亿）→ 必须检出"""
        c = _mutate(_good_collected(), 估值={"总股本": 1.39e8})
        m = extract_metrics(c)
        violations = check_against_golden(m, "301583")
        self.assertTrue(any("总股本" in v for v in violations))

    def test_wrong_annual_profit_flagged(self):
        """归母净利润错误 → 必须检出"""
        c = _mutate(_good_collected(), 利润表=[
            {"报告期": "一季报", "归母净利润": 0.1464e8},
            {"报告期": "年报", "归母净利润": 0.50e8},   # 0.9818 错写 0.50
            {"报告期": "一季报", "归母净利润": 0.1600e8},
        ])
        m = extract_metrics(c)
        violations = check_against_golden(m, "301583")
        self.assertTrue(any("2025年报归母" in v for v in violations))

    def test_stale_period_flagged(self):
        """财报时效滞后：最新报告期仅为三季报（应含一季报）→ 必须检出"""
        c = _mutate(_good_collected(), 利润表=[
            {"报告期": "三季报", "归母净利润": 0.2e8},
            {"报告期": "半年报", "归母净利润": 0.3e8},
            {"报告期": "年报", "归母净利润": 0.9818e8},
        ])
        m = extract_metrics(c)
        violations = check_against_golden(m, "301583")
        self.assertTrue(any("财报时效" in v for v in violations))

    def test_missing_required_field_flagged(self):
        """必需字段缺失 → 必须检出"""
        c = _good_collected()
        c["估值"] = {"data": []}
        m = extract_metrics(c)
        violations = check_against_golden(m, "301583")
        self.assertTrue(any("必需字段" in v for v in violations))

    def test_unknown_symbol(self):
        self.assertTrue(any("无黄金事实" in v for v in check_against_golden({"代码": "999999"}, "999999")))


class TestPipelineRunner(unittest.TestCase):
    """evaluate_data_pipeline 运行器 wiring：patch tools，走真实 collect_data 路径。"""

    def _mock_tools(self, good=True):
        from unittest.mock import patch

        def fake_fin(sym):
            if not good:
                return {"symbol": sym, "profit": [
                    {"报告期": "三季报", "归母净利润": 0.2e8},
                    {"报告期": "半年报", "归母净利润": 0.3e8},
                ]}
            return {"symbol": sym, "name": "托伦斯", "profit": [
                {"报告期": "一季报", "归母净利润": 0.1464e8, "扣非净利润": None},
                {"报告期": "年报", "归母净利润": 0.9818e8, "扣非净利润": None},
                {"报告期": "一季报", "归母净利润": 0.1600e8, "扣非净利润": None},
            ]}

        def fake_val(sym):
            if not good:
                return {"symbol": sym, "data": [{"总股本": 1.39e8, "每股净资产": 9.87}]}
            return {"symbol": sym, "data": [
                {"报告期": "2026-06-30 [半年报快报]", "总股本": 1.85473692e8,
                 "每股净资产": 9.87, "每股收益": 0.53}]}

        def fake_price(sym):
            return {"当前价格": 174.0, "name": "托伦斯"}

        def fake_anns(sym, count=20):
            return {"列表": [{"标题": "2026年半年度业绩快报", "日期": "2026-07-17",
                             "快报数据": {"报告期": "2026-06-30 [半年报快报]"}}]}

        return patch.multiple(
            "backend.tools",
            get_financial_statements=fake_fin,
            get_valuation=fake_val,
            fetch_stock_price=fake_price,
            get_recent_announcements=fake_anns,
        )

    def test_runner_passes_on_golden(self):
        from backend.eval_suite.data import evaluate_data_pipeline
        with self._mock_tools(good=True):
            report = evaluate_data_pipeline(["301583"])
        self.assertEqual(report["通过率"], 1.0)
        self.assertTrue(report["逐只"][0]["通过"])

    def test_runner_flags_data_errors(self):
        """注入数据错误（股本未除权 + 财报滞后）→ 通过率 0，违规非空"""
        from backend.eval_suite.data import evaluate_data_pipeline
        with self._mock_tools(good=False):
            report = evaluate_data_pipeline(["301583"])
        self.assertEqual(report["通过率"], 0.0)
        self.assertFalse(report["逐只"][0]["通过"])
        self.assertTrue(any("总股本" in v for v in report["逐只"][0]["违规"]))
        self.assertTrue(any("财报时效" in v for v in report["逐只"][0]["违规"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
