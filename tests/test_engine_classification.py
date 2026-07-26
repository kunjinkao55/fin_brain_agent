# -*- coding: utf-8 -*-
"""分类引擎修正方法（五支柱模型）回归测试

依据 docs/分类引擎修正方法.md：
支柱一 利润引擎解剖（锚定倾斜）/ 支柱二 SOTP强制隔离阀 / 支柱三 乘数矩阵 /
支柱四 驱动因子树情景（伪情景检测）/ 支柱五 逻辑一致性断路器（INV-7/9/10）。

全部 mock 离线运行。运行: python tests/test_engine_classification.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

# ---- 托伦斯 2025 年报真实主营构成（akshare stock_zygc_em 探查值） ----
TLS_SEGS = [
    {"名称": "半导体关键工艺零部件", "收入": 358439045.09, "收入占比": 0.497938,
     "毛利额": 121086386.41, "毛利占比": 0.619841, "毛利率": 0.337816},
    {"名称": "半导体结构零部件", "收入": 197999238.36, "收入占比": 0.275058,
     "毛利额": 42187943.83, "毛利占比": 0.215960, "毛利率": 0.213071},
    {"名称": "半导体工艺零部件", "收入": 92573194.12, "收入占比": 0.128601,
     "毛利额": 17352712.70, "毛利占比": 0.088828, "毛利率": 0.187449},
    {"名称": "激光设备零部件", "收入": 46320425.76, "收入占比": 0.064348,
     "毛利额": 8746112.71, "毛利占比": 0.044771, "毛利率": 0.188818},
    {"名称": "其他", "收入": 18536041.37, "收入占比": 0.02575,
     "毛利额": 777985.71, "毛利占比": 0.003983, "毛利率": 0.041972, "_其他": True},
    {"名称": "其中:匀气环", "收入": 106671400.0, "收入占比": 0.148186,
     "毛利额": 32581900.0, "毛利占比": 0.166787, "毛利率": 0.305442, "_子项": True},
]
TLS_PREV = [
    {"名称": "半导体关键工艺零部件", "收入": 304499718.80, "毛利率": 0.375154},
    {"名称": "半导体结构零部件", "收入": 173372167.50, "毛利率": 0.240789},
    {"名称": "半导体工艺零部件", "收入": 82485846.96, "毛利率": 0.208333},
    {"名称": "激光设备零部件", "收入": 36088902.58, "毛利率": 0.153563},
]

# ---- 东山精密式第二曲线 mock（老业务低增+新业务爆发） ----
DSJ_SEGS = [
    {"名称": "PCB老业务", "收入": 100e8, "收入占比": 0.80,
     "毛利额": 15e8, "毛利占比": 0.60, "毛利率": 0.15},
    {"名称": "光模块新业务", "收入": 25e8, "收入占比": 0.20,
     "毛利额": 10e8, "毛利占比": 0.40, "毛利率": 0.40},
]
DSJ_PREV = [
    {"名称": "PCB老业务", "收入": 95.2e8, "毛利率": 0.15},    # +5%
    {"名称": "光模块新业务", "收入": 13.9e8, "毛利率": 0.38},  # +80%
]


class TestEngineClassification(unittest.TestCase):
    """支柱一：利润引擎解剖与锚定倾斜"""

    def test_tls_anchor_tilts_to_dominant_engine(self):
        """托伦斯：半导体关键工艺零部件毛利占比 61.98%>40% → 锚定倾斜"""
        from backend.profit_engines import classify_engines, engine_anchor
        engines = classify_engines(TLS_SEGS, TLS_PREV)
        self.assertGreaterEqual(len(engines), 4)  # 子项/其他已剔除
        self.assertNotIn("匀气环", [e.name for e in engines])
        anchor = engine_anchor(engines)
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.name, "半导体关键工艺零部件")
        self.assertAlmostEqual(anchor.gp_pct, 0.619841, places=4)

    def test_dsj_second_curve_growth(self):
        """第二曲线：新业务 +80% 且毛利率不降 → 成长型；老业务 +5% → 非成长"""
        from backend.profit_engines import classify_engines
        engines = classify_engines(DSJ_SEGS, DSJ_PREV)
        by_name = {e.name: e for e in engines}
        self.assertEqual(by_name["光模块新业务"].engine_type, "成长型")
        self.assertNotEqual(by_name["PCB老业务"].engine_type, "成长型")

    def test_value_trap_detection(self):
        """增收+毛利率降>3pp → 价值陷阱型（托伦斯关键零部件 -3.7pp 命中）"""
        from backend.profit_engines import classify_engines
        engines = classify_engines(TLS_SEGS, TLS_PREV)
        key = next(e for e in engines if e.name == "半导体关键工艺零部件")
        self.assertEqual(key.engine_type, "价值陷阱型")

    def test_no_anchor_when_diversified(self):
        """板块分散（无>40%）→ 无锚定"""
        from backend.profit_engines import engine_anchor
        self.assertIsNone(engine_anchor(
            [type("E", (), {"gp_pct": 0.35})(), type("E", (), {"gp_pct": 0.30})()]))


class TestSOTPTriggers(unittest.TestCase):
    """支柱二：SOTP 强制隔离阀"""

    def test_dsj_triggers_both_valves(self):
        """东山精密式：增速差 75pp>30pp + 毛利率差 25pp>15pp → 双阀触发"""
        from backend.profit_engines import sotp_triggers
        reasons = sotp_triggers(DSJ_SEGS, DSJ_PREV)
        self.assertTrue(any("增速" in r for r in reasons))
        self.assertTrue(any("毛利率" in r for r in reasons))

    def test_tls_gm_gap_triggers(self):
        """托伦斯：毛利率极差 33.8%-18.7%=15.0pp+ >15pp → 毛利率阀触发（增速差不触发）"""
        from backend.profit_engines import sotp_triggers
        reasons = sotp_triggers(TLS_SEGS, TLS_PREV)
        self.assertTrue(any("毛利率" in r for r in reasons))

    def test_single_segment_no_trigger(self):
        """单一主业公司 → 不触发"""
        from backend.profit_engines import sotp_triggers
        segs = [{"名称": "唯一业务", "收入": 10e8, "收入占比": 1.0,
                 "毛利额": 2e8, "毛利占比": 1.0, "毛利率": 0.20}]
        self.assertEqual(sotp_triggers(segs, []), [])

    def test_sotp_fair_value_computation(self):
        """SOTP 估值：Σ(分配净利×板块PE)×(1-总部折价10%)"""
        from backend.profit_engines import classify_engines, sotp_fair_value
        engines = classify_engines(DSJ_SEGS, DSJ_PREV)
        total, detail = sotp_fair_value(engines, ttm_net=2e8, base_pe=30)
        self.assertGreater(total, 0)
        self.assertTrue(any("折价" in d for d in detail))
        # 总部折价 10% 已应用：total < Σ(未折价)
        self.assertLess(total, 2e8 * 30 * 1.5)


class TestIndicatorPeMatrix(unittest.TestCase):
    """支柱三：估值锚定匹配矩阵"""

    def test_bands(self):
        from backend.scoring import indicator_pe_matrix as m
        self.assertEqual(m(35, 1.0, 0.3)[1], 1.5)    # 三高档 → 高乘数
        self.assertEqual(m(5, -2.0, 1.5)[1], 0.6)    # 三低档 → 低乘数
        self.assertEqual(m(15, 0.0, 0.7)[1], 1.0)    # 混合 → 中乘数
        self.assertEqual(m(35, -2.0, 0.3)[1], 1.0)   # 毛利率低档否决高乘数
        self.assertEqual(m(None, None, None)[1], 1.0)  # 缺失 → 中性

    def test_matrix_path_in_rating(self):
        """matrix_mult>0 时替换 quality×growth，公式串三因子可复算"""
        import re
        from backend.scoring import compute_investment_rating
        d = compute_investment_rating(
            company_type="成长型",
            financial_scores={"盈利能力": {"得分": 5}, "成长性": {"得分": 8},
                              "财务健康": {"得分": 7}, "估值合理": {"得分": 5}},
            llm_scores={"行业前景": {"得分": 6}, "资金认可": {"得分": 5}},
            eps=1.0, stock_price=50.0, industry="半导体", roe=15, debt=40,
            matrix_mult=1.5, matrix_note="乘数矩阵[高乘数×1.5]",
        )
        chain = d["估值明细"]
        self.assertIn("乘数矩阵", chain)
        self.assertNotIn("财务质量乘数", chain)
        m = re.match(r'([\d.]+) × ([\d.]+) × ([\d.]+) = ([\d.]+)', chain["公式"])
        self.assertIsNotNone(m)
        eps_s, ind_pe, mm, fv = (float(g) for g in m.groups())
        self.assertAlmostEqual(eps_s * ind_pe * mm, fv, delta=0.01)

    def test_legacy_path_without_matrix(self):
        """不传 matrix_mult → 传统四因子路径（行为兼容）"""
        from backend.scoring import compute_investment_rating
        d = compute_investment_rating(
            company_type="成长型",
            financial_scores={"盈利能力": {"得分": 5}, "成长性": {"得分": 8},
                              "财务健康": {"得分": 7}, "估值合理": {"得分": 5}},
            llm_scores={"行业前景": {"得分": 6}, "资金认可": {"得分": 5}},
            eps=1.0, stock_price=50.0, industry="半导体", roe=15, debt=40,
        )
        self.assertIn("财务质量乘数", d["估值明细"])


class TestPseudoScenarioDetection(unittest.TestCase):
    """支柱四：PE 自变量伪情景检测"""

    def test_same_eps_all_scenarios_rebuilt(self):
        """三情景 EPS 相同仅 PE 不同 → 判定伪情景并按 TTM 带重建（EPS 恢复单调）"""
        from backend.agent import _validate_scenarios
        item = {"_eps_ttm": 0.5,
                "投资评级": {"当前价格": 50.0},
                "情景估值": {
                    "悲观": {"价格": 15.0, "EPS": 0.5, "PE": 30, "概率": "20%", "假设": "PE压缩"},
                    "基准": {"价格": 20.0, "EPS": 0.5, "PE": 40, "概率": "60%", "假设": "PE维持"},
                    "乐观": {"价格": 25.0, "EPS": 0.5, "PE": 50, "概率": "20%", "假设": "PE扩张"},
                }}
        _validate_scenarios(item)
        sc = item["情景估值"]
        self.assertTrue(any("伪情景" in n for n in item["_scenario_check"]["notes"]))
        # 重建后 EPS 恢复单调递增
        self.assertLess(sc["悲观"]["EPS"], sc["基准"]["EPS"])
        self.assertLess(sc["基准"]["EPS"], sc["乐观"]["EPS"])
        # 加权价值已重算
        self.assertIn("概率加权价值", sc)


class TestCircuitBreakers(unittest.TestCase):
    """支柱五：逻辑一致性断路器（INV-7/9/10）"""

    def test_inv7_sell_vs_high_growth(self):
        from backend.consistency import check_invariants
        item = {"投资评级": {"评级": "SELL", "合理价值": 10, "买入区间": "≤7元"},
                "_rev_growth": 35}
        self.assertTrue(any(i.rule == "INV-7" and i.severity == "warning"
                            for i in check_invariants(item)))
        item["_rev_growth"] = 20
        self.assertFalse(any(i.rule == "INV-7" for i in check_invariants(item)))

    def test_inv9_valuation_cashflow_mismatch(self):
        from backend.consistency import check_invariants
        item = {"投资评级": {"评级": "BUY", "合理价值": 100, "买入区间": "≤80元"},
                "_share_ctx": {"总股本": 1e8},   # 隐含市值 100亿
                "_cfo_annual": 2e8,              # 10×CFO = 20亿 < 100亿
                "_roic_proxy": 0.10}
        self.assertTrue(any(i.rule == "INV-9" and i.severity == "warning"
                            for i in check_invariants(item)))
        item["_roic_proxy"] = 0.20               # ROIC≥15% → 不触发
        self.assertFalse(any(i.rule == "INV-9" for i in check_invariants(item)))

    def test_inv10_cyclical_ttm_anchor(self):
        from backend.consistency import check_invariants
        item = {"投资评级": {"评级": "HOLD", "合理价值": 50, "买入区间": "≤35元"},
                "_cyclical": 0.5}
        self.assertTrue(any(i.rule == "INV-10" and i.severity == "warning"
                            for i in check_invariants(item)))
        item["_cyclical"] = 0.3
        self.assertFalse(any(i.rule == "INV-10" for i in check_invariants(item)))


def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestEngineClassification, TestSOTPTriggers, TestIndicatorPeMatrix,
                TestPseudoScenarioDetection, TestCircuitBreakers):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_all()
    sys.exit(0 if result.wasSuccessful() else 1)
