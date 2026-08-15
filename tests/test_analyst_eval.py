# -*- coding: utf-8 -*-
"""Layer B — analyst 推理质量评估的纯函数指标测试。

只测 evaluate_analysis 的纯函数（不调 LLM、不访问网络、不跑 RAG）：
    1. 数字幻觉率：输出含无出处硬数字 → 幻觉率高；仅可溯源数字 → 零幻觉
    2. 字段完整率：缺失关键字段 → 低于 100%
    3. 黄金事实命中：输出反映已知事实 → 命中；写错数值 → 未命中
    4. 结构化质量：JSON/评级/情景单调性

运行: python tests/test_analyst_eval.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

from backend.eval_suite.analyst import (
    GOLDEN_COLLECTED, GOLDEN_FACTS, REQUIRED_FIELDS,
    parse_analysis, hallucination_metrics, field_completeness,
    golden_fact_hits, structural_checks, evaluate_analysis,
)


# 与黄金事实一致的"合格"分析输出
def _good_analysis() -> str:
    import json
    return json.dumps({
        "代码": "301583",
        "名称": "托伦斯",
        "投资逻辑链": "精密制造龙头，次新+IPO（发行价22.6元），业绩快报增长",
        "评分": {"盈利能力": {"得分": 7, "依据": "ROE 11.78%，TTM EPS 0.516元"}},
        "亮点": ["次新股"], "风险": ["估值偏高"],
        "业绩驱动力": "精密制造订单放量",
        "关键信号": [{"信号": "PE高企", "数据": "337倍"}],
        "估值水位": {"PE": "337倍", "PB": "17.6", "市值": "322.7亿"},
        "情景估值": {
            "悲观": {"价格": 13.5, "EPS": 0.45, "PE": 30, "概率": "25%"},
            "基准": {"价格": 33.0, "EPS": 0.55, "PE": 60, "概率": "55%"},
            "乐观": {"价格": 52.0, "EPS": 0.65, "PE": 80, "概率": "20%"},
        },
        "催化剂": {"正面": ["业绩快报"], "负面": [], "强度": "中"},
        "证伪条件": ["毛利率下滑"],
        "操作建议": "当前174元估值偏贵，建议等待合理价值31.9元附近再评估",
        "投资评级": {"评级": "HOLD", "合理价值": 31.9, "当前价格": 174.0},
        "综合结论": {"总评": "基本面尚可但估值偏高"},
    }, ensure_ascii=False)


class TestParseAnalysis(unittest.TestCase):
    def test_parses_single_object(self):
        self.assertIsInstance(parse_analysis('{"代码": "301583"}'), dict)

    def test_parses_array_takes_first(self):
        self.assertEqual(parse_analysis('[{"代码": "A"}, {"代码": "B"}]')["代码"], "A")

    def test_fenced_json(self):
        self.assertIsInstance(parse_analysis('```json\n{"代码": "301583"}\n```'), dict)

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_analysis("这不是JSON"))


class TestHallucination(unittest.TestCase):
    def test_clean_output_zero_hallucination(self):
        """合格输出：所有硬数字都能在黄金 collected_data 中溯源 → 零幻觉"""
        good = _good_analysis()
        m = hallucination_metrics(good, GOLDEN_COLLECTED)
        self.assertEqual(m["幻觉率"], 0.0, f"合格输出出现幻觉: {m['无出处示例']}")

    def test_fabricated_number_detected(self):
        """注入编造硬数字（986倍，无出处）→ 幻觉率 > 0"""
        bad = _good_analysis().replace('"PE": "337倍"', '"PE": "986倍"')
        m = hallucination_metrics(bad, GOLDEN_COLLECTED)
        self.assertGreater(m["幻觉率"], 0.0)
        self.assertTrue(any("986" in s for s in m["无出处示例"]))

    def test_fabricated_target_price_detected(self):
        """注入编造目标价（88元，无出处）→ 无出处硬数字命中"""
        bad = _good_analysis().replace(
            "建议等待合理价值31.9元附近再评估",
            "建议等待合理价值31.9元附近再评估，目标价88元")
        m = hallucination_metrics(bad, GOLDEN_COLLECTED)
        self.assertTrue(any("88" in s for s in m["无出处示例"]))


class TestFieldCompleteness(unittest.TestCase):
    def test_complete_analysis_100(self):
        self.assertEqual(field_completeness(_good_analysis())["完整率"], 1.0)

    def test_missing_field_detected(self):
        import json
        data = json.loads(_good_analysis())
        del data["情景估值"]
        m = field_completeness(json.dumps(data, ensure_ascii=False))
        self.assertLess(m["完整率"], 1.0)
        self.assertIn("情景估值", m["缺失字段"])


class TestGoldenFacts(unittest.TestCase):
    def test_good_analysis_hits_all_facts(self):
        hits = golden_fact_hits(_good_analysis())
        self.assertEqual(hits["命中率"], 1.0, f"未命中: {hits['未命中']}")

    def test_wrong_numbers_missed(self):
        bad = _good_analysis().replace('"PB": "17.6"', '"PB": "88"')
        hits = golden_fact_hits(bad)
        self.assertIn("PB", hits["未命中"])


class TestStructural(unittest.TestCase):
    def test_good_analysis_passes(self):
        s = structural_checks(_good_analysis())
        self.assertTrue(s["json_ok"] and s["rating_ok"] and s["scenario_monotonic"])

    def test_inverted_scenarios_fail(self):
        import json
        data = json.loads(_good_analysis())
        data["情景估值"]["悲观"]["价格"] = 52.0  # 悲观 > 基准
        s = structural_checks(json.dumps(data, ensure_ascii=False))
        self.assertFalse(s["scenario_monotonic"])

    def test_garbage_fails_json(self):
        s = structural_checks("这不是JSON")
        self.assertFalse(s["json_ok"])


class TestEvaluateAnalysis(unittest.TestCase):
    def test_integration_good(self):
        r = evaluate_analysis(_good_analysis())
        self.assertEqual(r["数字幻觉"]["幻觉率"], 0.0)
        self.assertEqual(r["字段完整率"]["完整率"], 1.0)
        self.assertEqual(r["黄金事实"]["命中率"], 1.0)
        self.assertTrue(r["结构化"]["json_ok"])


class TestEvaluatorRunner(unittest.TestCase):
    """evaluate_analyst 运行器 wiring：patch LLM，走真实 analyst_node 路径。"""

    def _mock_factory(self, content: str):
        from unittest.mock import MagicMock

        class FakeResp:
            def __init__(self, c):
                self.content = c

            def model_dump_json(self, **kw):
                return self.content

        fake = MagicMock()
        fake.invoke.return_value = FakeResp(content)
        fake.with_structured_output.return_value = fake
        return fake

    def test_runner_produces_report(self):
        from unittest.mock import patch
        from backend.eval_suite.analyst import evaluate_analyst
        fake = self._mock_factory(_good_analysis())
        with patch("backend.agent._get_llm_with_schema", return_value=fake):
            report = evaluate_analyst("301583", runs=1)
        self.assertEqual(report["运行次数"], 1)
        self.assertEqual(report["失败次数"], 0)
        self.assertEqual(report["平均字段完整率"], 1.0)
        self.assertEqual(report["平均幻觉率"], 0.0)
        self.assertEqual(report["平均黄金事实命中率"], 1.0)

    def test_runner_survives_llm_error(self):
        from unittest.mock import patch, MagicMock
        from backend.eval_suite.analyst import evaluate_analyst
        fake = MagicMock()
        fake.with_structured_output.side_effect = RuntimeError("LLM down")
        # analyst_node 捕获后走 _raw_llm_fallback_with_retry（真实 LLM），一并 patch 避免网络挂起
        with patch("backend.agent._get_llm_with_schema", return_value=fake), \
             patch("backend.agent._raw_llm_fallback_with_retry",
                   side_effect=RuntimeError("fallback down")):
            report = evaluate_analyst("301583", runs=1)
        self.assertEqual(report["失败次数"], 1)
        self.assertIn("运行次数", report)
        self.assertNotIn("平均字段完整率", report)  # 无成功 run 时不应输出平均值


if __name__ == "__main__":
    unittest.main(verbosity=2)
