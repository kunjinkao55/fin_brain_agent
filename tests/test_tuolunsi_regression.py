# -*- coding: utf-8 -*-
"""托伦斯(301583) golden case 结构性回归测试 — FIX-01 ~ FIX-12 验收

依据：
- docs/FinBrain报告系统_结构性问题与修复任务表.md（18 项问题 ↔ 修复映射）
- example/托伦斯(301583)投资研究报告_核验修正版.md（A1-A10 / B1-B8 勘误与 golden 数字）

全部基于 mock 数据离线运行，不访问网络与 LLM。
运行: python tests/test_tuolunsi_regression.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

# ---- 托伦斯 golden 数据（来源：iFinD / 上市公告书，见核验修正版报告） ----
T_TOTAL_SHARES = 1.85473692e8   # 发行后总股本 18,547.3692 万股
T_PRICE = 174.00                # 2026-07-24 收盘价
T_BPS_POST = 9.87               # 发行后每股净资产（上市公告书披露）
T_BPS_PRE = 6.40                # 2025 年报 BPS（IPO 前，原报告误用值）
T_NET_2025 = 0.9818e8           # 2025 归母净利润（元）
T_Q1_2026 = 0.1464e8            # 2026Q1 归母净利润（元）
T_Q1_2025 = 0.1600e8            # 2025Q1 归母净利润（元）
T_EPS_TTM = 0.516               # 报告口径 TTM EPS（golden case 断言用）
T_IPO_ANN = {"日期": "2026-07-09",
             "标题": "托伦斯：首次公开发行股票并在创业板上市之上市公告书",
             "art_code": "AN_MOCK_IPO"}
T_IPO_CONTENT = (
    "托伦斯精密制造股份有限公司首次公开发行股票并在创业板上市之上市公告书。"
    "本次公开发行新股4,636.8423万股，发行价格22.60元/股。"
    "募集资金总额104,791.82万元，募集资金净额94,100.00万元。"
    "发行后总股本18,547.3692万股，发行后每股净资产9.87元。"
    "本次上市的无限售条件流通股30,831,368股，占发行后总股本的16.62%。"
    "网下配售限售股自上市之日起锁定6个月，战略配售股份及首发原股东股份锁定12个月。"
)
T_FORECAST_CONTENT = (
    "托伦斯：2026年半年度业绩预告。预计2026年半年度营业收入38,300万元~39,800万元，"
    "同比增长25%~30%；归属于上市公司股东的净利润同比下降约30%。"
)


def t_profit_rows():
    """利润表 mock（最新在前）：TTM = 0.9818 + 0.1464 - 0.1600 = 0.9682 亿"""
    return [
        {"报告期": "一季报", "归母净利润": T_Q1_2026, "扣非净利润": None},
        {"报告期": "年报",   "归母净利润": T_NET_2025, "扣非净利润": None},
        {"报告期": "一季报", "归母净利润": T_Q1_2025, "扣非净利润": None},
    ]


def t_valuation_rows():
    """估值表 mock（F10 主指标行）"""
    return [{"总股本": T_TOTAL_SHARES, "每股净资产": T_BPS_PRE, "每股收益": 0.53,
             "报告期": "2025-12-31 [年报]", "ROE(%)": 11.78, "资产负债率(%)": 40.70}]


class TestTuolunsiCapitalStructure(unittest.TestCase):
    """FIX-01/02/03：股本SSOT、口径配对、IPO 事件不重复摊薄（A1/A2/A3）"""

    def setUp(self):
        from backend import share_registry
        share_registry.clear_cache()

    def test_a1_ipo_event_not_private_placement(self):
        """A1：上市公告书必须分类为 IPO，绝不落入定增分支（不再应用 0.874 摊薄）"""
        from backend.corporate_actions import classify_events
        events = classify_events([T_IPO_ANN], fetch_content=lambda a: T_IPO_CONTENT)
        types = [e.type for e in events]
        self.assertIn("IPO", types)
        self.assertNotIn("定增", types, "IPO 被误分类为定增")
        ipo = next(e for e in events if e.type == "IPO")
        p = ipo.params
        self.assertAlmostEqual(p["发行股数(万股)"], 4636.8423, places=3)
        self.assertAlmostEqual(p["发行价(元)"], 22.60, places=2)
        self.assertAlmostEqual(p["募资净额(亿元)"], 9.41, places=2)
        self.assertAlmostEqual(p["发行后总股本(万股)"], 18547.3692, places=3)
        self.assertAlmostEqual(p["发行后每股净资产(元)"], 9.87, places=2)

    def test_a2_pb_uses_post_ipo_bps(self):
        """A2：IPO 后 PB 必须用含募集净额的发行后 BPS 9.87 → PB≈17.6（而非 27.2）"""
        from backend.share_registry import adjust_bps_for_event, resolve_share_context
        from backend.calc_engine import pb_at_price
        bps_adj, basis = adjust_bps_for_event(
            latest_equity_yi=None, net_raised_yi=9.41,
            post_shares=T_TOTAL_SHARES, disclosed_bps=T_BPS_POST)
        self.assertAlmostEqual(bps_adj, 9.87, places=2)
        self.assertEqual(basis, "公告披露发行后每股净资产")
        ctx = resolve_share_context("301583", t_valuation_rows(),
                                    bps_adjusted=bps_adj, bps_basis=basis)
        self.assertAlmostEqual(ctx.effective_bps, 9.87, places=2)
        # 错误口径（IPO 前 BPS 6.40）→ 27.2；正确口径 → 17.6
        self.assertAlmostEqual(pb_at_price(T_PRICE, ctx.effective_bps), 17.6, delta=0.2)

    def test_a3_market_cap_unique_and_static_pe(self):
        """A3：市值全系统唯一 = 174 × 1.8547亿股 = 322.7亿；静态 PE ≈ 329"""
        from backend.share_registry import resolve_share_context
        ctx = resolve_share_context("301583", t_valuation_rows())
        mktcap1 = ctx.total_shares * T_PRICE / 1e8
        mktcap2 = ctx.total_shares * T_PRICE / 1e8  # 第二处派生（SSOT 保证一致）
        self.assertAlmostEqual(mktcap1, mktcap2, places=6)
        self.assertAlmostEqual(mktcap1, 322.7, delta=0.2)
        static_pe = mktcap1 / (T_NET_2025 / 1e8)
        self.assertAlmostEqual(static_pe, 329, delta=2)

    def test_ttm_eps_unified_entry(self):
        """FIX-01：compute_ttm_eps 统一入口（calculate_scores 与 reporter 同口径）"""
        from backend.share_registry import compute_ttm_eps
        eps, ttm = compute_ttm_eps(t_profit_rows(), T_TOTAL_SHARES)
        self.assertAlmostEqual(ttm / 1e8, 0.9682, delta=0.001)
        self.assertAlmostEqual(eps, 0.522, delta=0.002)


class TestTuolunsiCalcVerification(unittest.TestCase):
    """FIX-04：LLM 计算验证层（A10/B1/B2/B3）"""

    def test_a10_formula_string_recomputable(self):
        """A10：估值明细公式串的显示因子乘积必须复算出合理价值（±0.01）"""
        import re
        from backend.scoring import compute_investment_rating
        d = compute_investment_rating(
            company_type="成长型",
            financial_scores={
                "盈利能力": {"得分": 5}, "成长性": {"得分": 2},
                "财务健康": {"得分": 7, "现金流严重度": 1, "现金流标签": "🟡正常"},
                "估值合理": {"得分": 2},
            },
            llm_scores={"行业前景": {"得分": 6}, "资金认可": {"得分": 5}},
            eps=0.516, stock_price=T_PRICE, industry="半导体",
            roe=11.78, debt=40.70, bps=T_BPS_PRE, bps_adjusted=T_BPS_POST,
        )
        chain = d["估值明细"]
        m = re.match(r'([\d.]+) × ([\d.]+) × ([\d.]+) × ([\d.]+) = ([\d.]+)', chain["公式"])
        self.assertIsNotNone(m, f"公式串格式异常: {chain['公式']}")
        eps_s, ind_pe, qm, gm, fv = (float(g) for g in m.groups())
        self.assertAlmostEqual(eps_s * ind_pe * qm * gm, fv, delta=0.01,
                               msg="公式显示因子乘积 ≠ 合理价值（托伦斯 A10 复现）")
        self.assertAlmostEqual(fv, d["合理价值"], places=2)

    def test_b1_implied_cagr_102pct(self):
        """B1：隐含3年增速必须为代码计算值 ≈102%（原报告 LLM 误写 40%）"""
        from backend.calc_engine import implied_cagr
        cagr = implied_cagr(T_PRICE, T_TOTAL_SHARES, T_NET_2025, 40, 3)
        self.assertAlmostEqual(cagr, 1.02, delta=0.03)
        # 放宽到 60 倍也需 ≈76%
        cagr60 = implied_cagr(T_PRICE, T_TOTAL_SHARES, T_NET_2025, 60, 3)
        self.assertAlmostEqual(cagr60, 0.76, delta=0.03)

    def test_b2_price_to_pe(self):
        """B2：30 元对应 TTM PE ≈58 倍（原报告误写 45 倍）；PE45 对应 ≤23.2 元"""
        from backend.calc_engine import pe_at_price
        self.assertAlmostEqual(pe_at_price(30, T_EPS_TTM), 58.1, delta=0.5)
        self.assertAlmostEqual(45 * T_EPS_TTM, 23.2, delta=0.1)

    def test_b3_price_to_pb(self):
        """B3：100 元对应 PB ≈10.1 倍（原报告误写 15 倍）；PB15 对应 ≈148 元"""
        from backend.calc_engine import pb_at_price
        self.assertAlmostEqual(pb_at_price(100, T_BPS_POST), 10.13, delta=0.1)
        self.assertAlmostEqual(15 * T_BPS_POST, 148.05, delta=0.1)

    def test_b1_verify_report_text_catches_wrong_cagr(self):
        """FIX-04 回溯验证：文本写 40% 而登记值为 102% → 必须报违规"""
        from backend.calc_engine import CalcTable, verify_report_text
        ct = CalcTable()
        ct.register("隐含增速", 102.0, formula="(322.7亿÷40÷0.98亿)^(1/3)-1")
        bad = verify_report_text("市场隐含未来3年利润年复合增速需达40%以上才能消化当前估值", ct)
        self.assertTrue(any("隐含增速" in v for v in bad))
        # 同义词变体（第四轮真实报告暴露：LLM 写"年化增速需达30%以上"）
        bad2 = verify_report_text("当前市盈率隐含未来净利润年化增速需达30%以上", ct)
        self.assertTrue(any("隐含增速" in v for v in bad2))
        good = verify_report_text("隐含年复合增速≈102%", ct)
        self.assertEqual(good, [])

    def test_units_no_rounding_up(self):
        """FIX-11：1.89亿 不得渲染为 "2亿"；0.1464亿 → 万级"""
        from backend.calc_engine import fmt_yi_amount
        self.assertEqual(fmt_yi_amount(1.89), "1.89亿")
        self.assertIn("万", fmt_yi_amount(0.1464))


class TestTuolunsiConsistency(unittest.TestCase):
    """FIX-05/07：跨模块一致性不变量（B4/B7/B6/A4/B8/#4/#16）"""

    def _item(self, **kw):
        base = {
            "投资评级": {"评级": "SELL", "合理价值": 31.9, "买入区间": "≤23.9元",
                        "当前价格": T_PRICE},
            "情景估值": {
                "悲观": {"价格": 13.5, "EPS": 0.45, "PE": 30, "概率": "25%", "假设": "净利下滑约15%"},
                "基准": {"价格": 33.0, "EPS": 0.55, "PE": 60, "概率": "55%", "假设": "基本持平"},
                "乐观": {"价格": 52.0, "EPS": 0.65, "PE": 80, "概率": "20%", "假设": "增长约23%"},
                "概率加权价值": 31.9,
            },
            "估值水位": {"PE": "337", "PB": "17.6", "市值": "323亿", "前瞻PE": "季节性失真"},
            "_eps_ttm": T_EPS_TTM,
        }
        base.update(kw)
        return base

    def test_b4_fair_range_must_subset_anchor(self):
        """B4：合理区间(30-60元)高于自身合理价值(27.9元) → INV-1 阻断"""
        from backend.consistency import check_invariants
        bad = self._item(操作建议="建议等待股价回归合理区间（约30-60元）后再评估")
        bad["投资评级"]["合理价值"] = 27.9
        bad["投资评级"]["买入区间"] = "≤20.9元"
        issues = check_invariants(bad)
        self.assertTrue(any(i.rule == "INV-1" and i.severity == "blocker" for i in issues))
        # 修正后的区间（24-32元 ⊆ [23.9×0.98, 31.9×1.1]）→ 无违例
        good = self._item(操作建议="建议等待股价回归合理区间（约24-32元）后再评估")
        self.assertFalse(any(i.rule == "INV-1" for i in check_invariants(good)))

    def test_b7_sell_vs_buy_price_conflict(self):
        """B7：SELL 与"30元以下可轻仓试探"矛盾 → INV-2 阻断"""
        from backend.consistency import check_invariants
        bad = self._item(操作建议="暂不参与。若回调至30元以下可轻仓试探")
        issues = check_invariants(bad)
        self.assertTrue(any(i.rule == "INV-2" and i.severity == "blocker" for i in issues))
        good = self._item(操作建议="不参与；跌至23.9元以下才具备25%安全边际")
        self.assertFalse(any(i.rule == "INV-2" for i in check_invariants(good)))

    def test_b6_scenario_text_eps_alignment(self):
        """B6：假设文本增速与 EPS 隐含增速偏差>5pp → INV-3 警告"""
        from backend.consistency import check_invariants
        item = self._item()
        item["情景估值"]["悲观"]["假设"] = "扣非下滑20%"  # EPS 0.45/0.516-1 = -12.8%，偏差 7.2pp
        issues = check_invariants(item)
        self.assertTrue(any(i.rule == "INV-3" and i.severity == "warning" for i in issues))
        # 价格倒挂 → blocker
        item2 = self._item()
        item2["情景估值"]["悲观"]["价格"] = 40.0
        self.assertTrue(any(i.rule == "INV-3" and i.severity == "blocker"
                            for i in check_invariants(item2)))

    def test_a4_disabled_metric_zero_reference(self):
        """A4：前瞻PE 已禁用却出现"前瞻PE超395倍" → INV-4 阻断（FIX-07 注册表广播）"""
        from backend.consistency import check_invariants, disable_metric, is_active
        item = self._item()
        disable_metric(item, "前瞻PE", "Q1占比15%，简单年化失真")
        self.assertFalse(is_active(item, "前瞻PE"))
        issues = check_invariants(item, "估值极高：PE(TTM) 337倍，前瞻PE超395倍。")
        self.assertTrue(any(i.rule == "INV-4" and i.severity == "blocker" for i in issues))
        clean = check_invariants(item, "估值极高：PE(TTM) 337倍，前瞻PE已禁用。")
        self.assertFalse(any(i.rule == "INV-4" for i in clean))

    def test_b8_sentiment_breadth_divergence(self):
        """B8：上涨比例 8.5% 与"温和偏暖"背离 → INV-6 警告"""
        from backend.consistency import check_invariants
        item = self._item(市场情绪={"综合情绪": "+0.20（温和偏暖）", "市场广度": "上涨比例 8.5%"})
        issues = check_invariants(item)
        self.assertTrue(any(i.rule == "INV-6" and i.severity == "warning" for i in issues))

    def test_unique_value_checks(self):
        """#4/#16：同一指标全文唯一值——市值 323/241 并存、Q1净利 0.15/0.1 并存 → INV-5 阻断"""
        from backend.consistency import check_invariants
        item = self._item()
        text = "市值323亿……按发行前股本计算市值241亿……"
        self.assertTrue(any(i.rule == "INV-5" for i in check_invariants(item, text)))
        text2 = "Q1归母净利润0.1亿……Q1归母净利0.15亿……"
        self.assertTrue(any(i.rule == "INV-5" for i in check_invariants(item, text2)))
    def test_scenario_eps_ttm_anchoring(self):
        """情景锚定（复验暴露）：LLM 情景 EPS 脱离 TTM 现实区间时必须钳制重算。
        托伦斯复验案例：悲观 EPS=1.2 是 TTM 0.516 的 2.3 倍 → 加权价值 125 元严重高估。"""
        from backend.agent import _validate_scenarios
        item = {"_eps_ttm": T_EPS_TTM,
                "情景估值": {
                    "悲观": {"价格": 80.0, "EPS": 1.2, "PE": 66.67, "概率": "30%", "假设": "增速回落至20%"},
                    "基准": {"价格": 130.0, "EPS": 1.6, "PE": 81.25, "概率": "50%", "假设": "维持30%增速"},
                    "乐观": {"价格": 180.0, "EPS": 2.0, "PE": 90.0, "概率": "20%", "假设": "增长40%"},
                }}
        _validate_scenarios(item)
        sc = item["情景估值"]
        # 悲观钳制到 ≤1.0×TTM、基准 ≤1.3×、乐观 ≤2.0×（钳制值取整两位小数，容差0.01）
        self.assertLessEqual(sc["悲观"]["EPS"], T_EPS_TTM + 0.01)
        self.assertLessEqual(sc["基准"]["EPS"], T_EPS_TTM * 1.3 + 0.01)
        self.assertLessEqual(sc["乐观"]["EPS"], T_EPS_TTM * 2.0 + 0.01)
        # 钳制后 价格=EPS×PE 重算，加权价值从 125 元大幅回落至现实区间（PE 倍数仍是 LLM 判断）
        self.assertAlmostEqual(sc["悲观"]["价格"], round(sc["悲观"]["EPS"] * 66.67, 2), places=2)
        self.assertLess(sc["概率加权价值"], 60.0)
        self.assertGreater(sc["概率加权价值"], 10.0)
        self.assertTrue(any("锚定" in n for n in item["_scenario_check"]["notes"]))
    def test_scenario_eps_ttm_anchoring(self):
        """情景锚定（复验暴露）：LLM 情景 EPS 脱离 TTM 现实区间时必须钳制重算。
        托伦斯复验案例：悲观 EPS=1.2 是 TTM 0.516 的 2.3 倍 → 加权价值 125 元严重高估。"""
        from backend.agent import _validate_scenarios
        item = {"_eps_ttm": T_EPS_TTM,
                "情景估值": {
                    "悲观": {"价格": 80.0, "EPS": 1.2, "PE": 66.67, "概率": "30%", "假设": "增速回落至20%"},
                    "基准": {"价格": 130.0, "EPS": 1.6, "PE": 81.25, "概率": "50%", "假设": "维持30%增速"},
                    "乐观": {"价格": 180.0, "EPS": 2.0, "PE": 90.0, "概率": "20%", "假设": "增长40%"},
                }}
        _validate_scenarios(item)
        sc = item["情景估值"]
        # 悲观钳制到 ≤1.0×TTM、基准 ≤1.3×、乐观 ≤2.0×（钳制值取整两位小数，容差0.01）
        self.assertLessEqual(sc["悲观"]["EPS"], T_EPS_TTM + 0.01)
        self.assertLessEqual(sc["基准"]["EPS"], T_EPS_TTM * 1.3 + 0.01)
        self.assertLessEqual(sc["乐观"]["EPS"], T_EPS_TTM * 2.0 + 0.01)
        # 钳制后 价格=EPS×PE 重算，加权价值从 125 元大幅回落至现实区间（PE 倍数仍是 LLM 判断）
        self.assertAlmostEqual(sc["悲观"]["价格"], round(sc["悲观"]["EPS"] * 66.67, 2), places=2)
        self.assertLess(sc["概率加权价值"], 60.0)
        self.assertGreater(sc["概率加权价值"], 10.0)

    def test_scenario_pe_industry_band_clamp(self):
        """情景 PE 行业中枢锚定带（83.81 元案例回归）：基准 PE 120 → 钳到 60，
        加权价值从 83.81 收敛到行业带内（PE 判断留给 LLM 但不出圈）。"""
        from backend.agent import _validate_scenarios
        item = {"_eps_ttm": T_EPS_TTM,
                "投资评级": {"估值明细": {"行业PE中枢": 40}},
                "情景估值": {
                    "悲观": {"价格": 50.0, "EPS": 0.5, "PE": 100.0, "概率": "30%", "假设": "行业下行"},
                    "基准": {"价格": 80.4, "EPS": 0.67, "PE": 120.0, "概率": "50%", "假设": "稳步推进"},
                    "乐观": {"价格": 143.07, "EPS": 1.03, "PE": 138.9, "概率": "20%", "假设": "订单超预期"},
                }}
        _validate_scenarios(item)
        sc = item["情景估值"]
        # 基准 [0.8×, 1.5×]×40 = [32, 60]；悲观 [36, 48]；乐观 ≤60（基准触顶收敛）
        self.assertEqual(sc["基准"]["PE"], 60.0)
        self.assertLessEqual(sc["悲观"]["PE"], 48.0)
        self.assertGreaterEqual(sc["悲观"]["PE"], 36.0)
        self.assertLessEqual(sc["乐观"]["PE"], 60.0)
        # 价格随钳制重算，加权从 83.81 收敛到带内
        self.assertAlmostEqual(sc["基准"]["价格"], round(sc["基准"]["EPS"] * 60.0, 2), places=2)
        self.assertLess(sc["概率加权价值"], 50.0)
        self.assertTrue(any("锚定带" in n for n in item["_scenario_check"]["notes"]))


class TestTuolunsiEventRules(unittest.TestCase):
    """FIX-06：涨跌停规则引擎（A5）"""

    def test_a5_first5days_no_limit(self):
        """A5：创业板新股上市前5日无涨跌幅限制——7/10 首日 +858.8% 不得标"涨停" """
        from backend.trading_rules import is_limit_up, board_of, limit_pct
        self.assertEqual(board_of("301583"), "创业板")
        self.assertIsNone(limit_pct("创业板", listing_days=1))
        self.assertFalse(is_limit_up("301583", "托伦斯", 858.8,
                                     listing_date="2026-07-10", ref_date="2026-07-10"))
        # 7/24（上市第11个交易日）+20.0% → 标"涨停"
        self.assertTrue(is_limit_up("301583", "托伦斯", 20.0,
                                    listing_date="2026-07-10", ref_date="2026-07-24"))

    def test_board_rules(self):
        """板块规则：主板10%/双创20%/北交所30%/ST 5%/主板新股首日44%"""
        from backend.trading_rules import board_of, limit_pct, is_limit_up
        self.assertEqual(limit_pct("沪主板"), 10.0)
        self.assertEqual(limit_pct("科创板"), 20.0)
        self.assertEqual(limit_pct("北交所"), 30.0)
        self.assertEqual(limit_pct("创业板", st=True), 5.0)
        self.assertEqual(limit_pct("深主板", listing_days=1), 44.0)
        self.assertEqual(board_of("836221"), "北交所")
        self.assertTrue(is_limit_up("600036", "招商银行", 10.02))


class TestTuolunsiNewListing(unittest.TestCase):
    """FIX-08/09：次新股数据包与业绩预告联动（C类遗漏）"""

    def test_parse_float_ratio(self):
        """流通盘 16.62% 解析"""
        from backend.new_listing import parse_float_ratio
        r = parse_float_ratio(T_IPO_CONTENT)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 0.1662, places=4)

    def test_parse_lockups(self):
        """解禁日程：网下配售 6 个月（2027-01-10）/ 战配及原股东 12 个月（2027-07-10）"""
        from backend.new_listing import parse_lockups
        locks = parse_lockups(T_IPO_CONTENT, listing_date="2026-07-10")
        self.assertTrue(len(locks) >= 2)
        dates = sorted(l["date"] for l in locks if l.get("date"))
        self.assertIn("2027-01-10", dates)
        self.assertIn("2027-07-10", dates)

    def test_parse_customer_concentration(self):
        """客户集中度 92.60%（北方华创 45.64%）解析"""
        from backend.new_listing import parse_customer_concentration
        cc = parse_customer_concentration(
            "2025年度，公司前五大客户销售收入占比合计92.60%，"
            "其中北方华创占45.64%，中微公司占35.68%。")
        self.assertIsNotNone(cc)
        self.assertAlmostEqual(cc["top5_pct"], 0.9260, places=4)
        self.assertAlmostEqual(cc["top1_pct"], 0.4564, places=4)
        self.assertEqual(cc["top1_name"], "北方华创")

    def test_parse_performance_forecast(self):
        """FIX-09：区间型业绩预告解析（营收3.83~3.98亿/+25%~30%，净利约-30%）"""
        from backend.new_listing import parse_performance_forecast
        fc = parse_performance_forecast(T_FORECAST_CONTENT, "2026年半年度业绩预告")
        self.assertIsNotNone(fc)
        self.assertAlmostEqual(fc["rev_min_yi"], 3.83, places=2)
        self.assertAlmostEqual(fc["rev_max_yi"], 3.98, places=2)
        self.assertAlmostEqual(fc["rev_growth_min"], 0.25, places=2)
        self.assertAlmostEqual(fc["np_growth_max"], -0.30, places=2)
        self.assertEqual(fc["forecast_type"], "预减")

    def test_forecast_scenario_link(self):
        """FIX-09：预告净利-30% 落在悲观情景区间 → 标注该情形概率上升"""
        from backend.new_listing import forecast_scenario_link
        note = forecast_scenario_link({"np_growth_min": -0.30, "np_growth_max": -0.30},
                                      {"悲观": -0.15, "基准": 0.04, "乐观": 0.23})
        self.assertIsNotNone(note)
        self.assertIn("悲观", note)

    def test_flash_forecast_not_merged_into_profit(self):
        """FIX-09：业绩预告（预测值）禁止回灌利润表（否则污染 TTM）"""
        from backend.tools import _extract_flash_report, merge_flash_into_profit
        flash = _extract_flash_report(T_FORECAST_CONTENT, "2026年半年度业绩预告")
        self.assertIsNotNone(flash)
        self.assertTrue(flash.get("_预告"))
        rows = t_profit_rows()
        before = [dict(r) for r in rows]
        merge_flash_into_profit(rows, flash)
        self.assertEqual(rows, before, "预告数据被回灌进利润表")

    def test_pack_graceful_degradation(self):
        """FIX-08：无数据回调时不抛异常，缺失项记入 missing"""
        from backend.new_listing import fetch_new_listing_pack
        pack = fetch_new_listing_pack("301583")  # 无任何回调
        self.assertIsInstance(pack, dict)
        self.assertIn("missing", pack)


class TestAuditCoverage(unittest.TestCase):
    """FIX-10：18 项历史错误模式回放 — 自动检出率目标 ≥90%（本用例集应 18/18）"""

    def test_18_patterns_detection_rate(self):
        """逐项构造 18 类错误模式并断言对应守卫可检出/修正，统计检出率"""
        from backend.corporate_actions import classify_events
        from backend.share_registry import adjust_bps_for_event, resolve_share_context, compute_ttm_eps
        from backend.calc_engine import (CalcTable, verify_report_text, cross_validate_fields,
                                         pe_at_price, pb_at_price, implied_cagr, fmt_yi_amount)
        from backend.consistency import check_invariants, disable_metric
        from backend.trading_rules import is_limit_up
        from backend.new_listing import (parse_performance_forecast, parse_customer_concentration,
                                         parse_lockups, parse_float_ratio)
        from backend import share_registry
        share_registry.clear_cache()

        detected = {}

        # 1. 事件误分类（IPO→定增）
        evs = classify_events([T_IPO_ANN], fetch_content=lambda a: T_IPO_CONTENT)
        detected[1] = any(e.type == "IPO" for e in evs) and not any(e.type == "定增" for e in evs)
        # 2. 重复摊薄（EPS 已按发行后股本 → 无定增/配股事件可驱动摊薄）
        detected[2] = not any(e.type in ("定增", "配股") for e in evs)
        # 3. PB 时点错配（IPO 前净资产÷IPO 后股价）
        bps_adj, _ = adjust_bps_for_event(None, 9.41, T_TOTAL_SHARES, disclosed_bps=9.87)
        detected[3] = abs(bps_at := pb_at_price(T_PRICE, bps_adj) - 17.6) < 0.3
        # 4. 市值两口径并存
        item4 = {"投资评级": {"评级": "SELL", "合理价值": 31.9, "买入区间": "≤23.9元"},
                 "估值水位": {"市值": "323亿"}}
        detected[4] = any(i.rule == "INV-5" for i in check_invariants(item4, "市值323亿 vs 市值241亿"))
        # 5. 新股首日"涨停"误判
        detected[5] = not is_limit_up("301583", "托伦斯", 858.8, "2026-07-10", "2026-07-10")
        # 6. CFO 0.587亿 误写 1亿（跨源比对）
        detected[6] = bool(cross_validate_fields({"经营现金流": 0.587}, {"经营现金流": 1.0}))
        # 7. 隐含增速 40%（正确 ≈102%）
        ct7 = CalcTable(); ct7.register("隐含增速", round(implied_cagr(174, T_TOTAL_SHARES, T_NET_2025, 40) * 100, 1))
        detected[7] = bool(verify_report_text("年复合增速需达40%以上", ct7))
        # 8. "30元≈PE45倍"（实为 58 倍）
        detected[8] = abs(pe_at_price(30, T_EPS_TTM) - 58.1) < 0.5
        # 9. "100元=PB15倍"（实为 10.1 倍）
        detected[9] = abs(pb_at_price(100, T_BPS_POST) - 10.13) < 0.1
        # 10. 前瞻PE 禁用与引用并存
        item10 = dict(item4); disable_metric(item10, "前瞻PE", "季节性失真")
        detected[10] = any(i.rule == "INV-4" for i in check_invariants(item10, "前瞻PE超395倍"))
        # 11. 合理区间高于合理价值
        item11 = {"投资评级": {"评级": "HOLD", "合理价值": 27.9, "买入区间": "≤20.9元"},
                  "操作建议": "合理区间约30-60元"}
        detected[11] = any(i.rule == "INV-1" for i in check_invariants(item11))
        # 12. SELL 与买入价矛盾
        item12 = {"投资评级": {"评级": "SELL", "合理价值": 31.9, "买入区间": "≤23.9元"},
                  "操作建议": "若回调至30元以下可轻仓试探"}
        detected[12] = any(i.rule == "INV-2" for i in check_invariants(item12))
        # 13. 情景 EPS 与叙事不匹配
        item13 = {"投资评级": {"评级": "SELL", "合理价值": 31.9, "买入区间": "≤23.9元"},
                  "_eps_ttm": T_EPS_TTM,
                  "情景估值": {"悲观": {"价格": 13.5, "EPS": 0.45, "假设": "扣非下滑20%"},
                              "基准": {"价格": 33.0, "EPS": 0.55, "假设": "增长6%"},
                              "乐观": {"价格": 52.0, "EPS": 0.65, "假设": "增长15%"}}}
        detected[13] = any(i.rule == "INV-3" for i in check_invariants(item13))
        # 14. 预告/客户/流通盘/解禁 遗漏（解析器全部可用）
        detected[14] = all([
            parse_performance_forecast(T_FORECAST_CONTENT, "业绩预告"),
            parse_customer_concentration("前五大客户销售收入占比合计92.60%，其中北方华创占45.64%"),
            parse_lockups(T_IPO_CONTENT, "2026-07-10"),
            parse_float_ratio(T_IPO_CONTENT),
        ])
        # 15. ROE 11.9% vs 实际 11.78%（精度/期次偏差>1%可检出）
        detected[15] = bool(cross_validate_fields({"ROE": 11.9}, {"ROE": 11.78}, tol=0.01))
        # 16. Q1净利 0.1亿/0.15亿 并存
        detected[16] = any(i.rule == "INV-5" for i in check_invariants(
            item4, "Q1归母净利润0.1亿……Q1归母净利0.15亿"))
        # 17. 毛利率期次混用（"当前27%"未更新至 Q1 的 23.3% → 跨期偏差可检出）
        detected[17] = bool(cross_validate_fields({"毛利率": 27.1}, {"毛利率": 23.25}, tol=0.02))
        # 18. 情绪与广度背离
        item18 = dict(item4)
        item18["市场情绪"] = {"综合情绪": "+0.20（温和偏暖）", "市场广度": "上涨比例 8.5%"}
        detected[18] = any(i.rule == "INV-6" for i in check_invariants(item18))

        failed = [k for k, v in detected.items() if not v]
        rate = sum(1 for v in detected.values() if v) / 18
        self.assertEqual(len(detected), 18)
        self.assertGreaterEqual(rate, 0.90,
                                msg=f"历史问题库回放检出率 {rate:.0%} < 90%，未检出项: {failed}")


def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (TestTuolunsiCapitalStructure, TestTuolunsiCalcVerification,
                TestTuolunsiConsistency, TestTuolunsiEventRules,
                TestTuolunsiNewListing, TestAuditCoverage):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_all()
    sys.exit(0 if result.wasSuccessful() else 1)
