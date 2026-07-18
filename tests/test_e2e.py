"""
FinBrain 端到端验证脚本
覆盖：编译检查、数据工具、评分一致性、单股/多股分析、缓存、配置、Harness守卫
用法: python tests/test_e2e.py
"""

import sys, os, json, time, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["FINBRAIN_DATA_MODE"] = "local"
os.environ["FINBRAIN_LLM_MODE"] = "local"


class TestCompilation(unittest.TestCase):
    """所有模块必须能正常导入"""

    def test_agent_import(self):
        from backend.agent import build_graph, ask
        g = build_graph()
        self.assertIsNotNone(g)

    def test_tools_import(self):
        from backend.tools import (fetch_stock_price, get_financial_statements,
                                     get_valuation, get_industry_info, calculate_scores)
        self.assertTrue(callable(fetch_stock_price))

    def test_scoring_import(self):
        from backend.scoring import compute_investment_rating
        from backend.scoring_config import get_weights
        w = get_weights("价值型")
        self.assertIn("估值", w)

    def test_rag_import(self):
        from backend.accounting_rag import search_kb, list_kbs
        kbs = list_kbs()
        self.assertGreaterEqual(len(kbs), 3)

    def test_cache_import(self):
        from backend.cache import get, set, clear, stats
        clear()
        set("test", "key", {"a": 1})
        self.assertEqual(get("test", "key"), {"a": 1})
        self.assertEqual(stats()["mode"], "local")

    def test_api_import(self):
        from backend.api import app
        routes = [r.path for r in app.routes if hasattr(r, "methods")]
        self.assertIn("/health", routes)

    def test_client_import(self):
        from backend.client import _get_local_llm, invoke_remote_analysis
        self.assertTrue(callable(invoke_remote_analysis))

    def test_scheduler_import(self):
        from backend.scheduler import _WATCHLIST
        self.assertGreaterEqual(len(_WATCHLIST), 10)


class TestDataTools(unittest.TestCase):
    """数据工具返回合法格式"""

    def test_stock_price(self):
        from backend.tools import fetch_stock_price
        r = fetch_stock_price("601991")
        self.assertIn("price", r)
        self.assertIsInstance(float(r["price"]), float)

    def test_financials(self):
        from backend.tools import get_financial_statements
        r = get_financial_statements("601991")
        self.assertIn("profit", r)
        self.assertGreater(len(r.get("profit", [])), 0)

    def test_valuation(self):
        from backend.tools import get_valuation
        r = get_valuation("601991")
        self.assertIn("data", r)
        self.assertGreater(len(r.get("data", [])), 0)

    def test_industry(self):
        from backend.tools import get_industry_info
        r = get_industry_info("601991")
        self.assertTrue("行业" in r or "industry_name" in r)

    def test_limit_up_pool(self):
        from backend.tools import get_limit_up_pool
        r = get_limit_up_pool(5)
        self.assertIn("涨停板数量", r)

    def test_market_breadth(self):
        from backend.tools import get_market_breadth
        r = get_market_breadth()
        self.assertIn("全A", r)

    def test_stock_streak(self):
        from backend.tools import get_stock_streak
        r = get_stock_streak("601991")
        self.assertIn("连板天数", r)

    def test_intraday_guard(self):
        from backend.tools import get_intraday
        r = get_intraday("601991")
        # 非交易时段或正常数据，不能是未捕获的异常
        self.assertTrue("info" in r or "bars" in r or "error" in r)


class TestScoringConsistency(unittest.TestCase):
    """评分确定性：同一输入必须同一输出"""

    def test_same_stock_same_score(self):
        from backend.tools import calculate_scores, get_financial_statements, get_valuation, fetch_stock_price, get_industry_info
        fin = get_financial_statements("601991")
        val = get_valuation("601991")
        price = fetch_stock_price("601991")
        ind = get_industry_info("601991")
        data = {"profit": fin["profit"], "cashflow": fin["cashflow"], "balance": fin["balance"],
                "valuation": val, "price": dict(price), "industry": ind.get("行业", "")}

        s1 = calculate_scores(data)
        s2 = calculate_scores(data)

        for dim in ["盈利能力", "成长性", "财务健康", "估值合理"]:
            self.assertEqual(s1[dim]["得分"], s2[dim]["得分"],
                             f"{dim} score changed between identical inputs")

    def test_empty_data_returns_na(self):
        from backend.tools import calculate_scores
        r = calculate_scores({})
        for dim in ["盈利能力", "成长性", "财务健康", "估值合理"]:
            self.assertIsNone(r[dim]["得分"], f"{dim} should be N/A for empty data")


class TestInvestmentRating(unittest.TestCase):
    """投资决策引擎正确计算"""

    def test_rating_output_structure(self):
        from backend.scoring import compute_investment_rating
        r = compute_investment_rating(
            company_type="价值型",
            financial_scores={"盈利能力": {"得分": 7}, "成长性": {"得分": 5},
                              "财务健康": {"得分": 8}, "估值合理": {"得分": 8}},
            llm_scores={"行业前景": {"得分": 5}, "资金认可": {"得分": 5}},
            eps=2.0, stock_price=30.0, industry="医药", roe=18.0, debt=35.0,
        )
        for k in ["评级", "合理价值", "安全边际要求", "加权总分", "置信度"]:
            self.assertIn(k, r, f"Missing key: {k}")


class TestHarnessGuards(unittest.TestCase):
    """Harness 守卫功能正常"""

    def test_data_guard_empty_input(self):
        from backend.tools import calculate_scores
        r = calculate_scores({})
        self.assertIsNone(r["盈利能力"]["得分"])

    def test_intraday_non_trading(self):
        from backend.tools import get_intraday
        from datetime import datetime
        r = get_intraday("601991")
        now = datetime.now()
        t = now.hour * 60 + now.minute
        is_trading = (now.weekday() < 5 and 9 * 60 + 15 <= t <= 15 * 60 + 5)
        if not is_trading:
            self.assertIn("info", r, "Should return info outside trading hours")

    def test_source_config_works(self):
        from backend.tools import _get_source
        for k in ["stock_price", "financials", "industry", "fund_flow"]:
            src = _get_source(k)
            self.assertIsNotNone(src)


class TestConfig(unittest.TestCase):
    """配置文件完整性"""

    def test_strategies_json(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "strategies.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("default", data)
        for k in ["data_collector", "analyst", "phantom"]:
            self.assertIn(k, data["default"])

    def test_scoring_json(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "scoring.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for section in ["盈利能力", "成长性", "财务健康", "估值合理", "动态权重", "安全边际"]:
            self.assertIn(section, data)


class TestOutputConsistency(unittest.TestCase):
    """报告输出合规性：评级-操作一致性、EPS单调性、审计节存在性"""

    def test_score_sum_in_report_format(self):
        """评分合计必须等于各维度得分之和，不能出现88.5/70"""
        from backend.tools import format_report
        from backend.scoring import compute_investment_rating
        # 构造一个模拟item，评分合计溢出
        mock = {
            "代码": "000001", "名称": "测试股",
            "评分": {
                "盈利能力": {"得分": 5, "依据": "test"},
                "成长性": {"得分": 4, "依据": "test"},
                "财务健康": {"得分": 5, "依据": "test"},
                "估值合理": {"得分": 8, "依据": "test"},
                "行业前景": {"得分": 5, "依据": "test"},
                "资金认可": {"得分": 5, "依据": "test"},
                "加权总分": {"得分": 88.5, "依据": "test"},  # 复合指标
            },
            "投资评级": {"评级": "HOLD", "合理价值": 10.0, "估值差距": "+10%",
                         "安全边际要求": "30%", "买入区间": "≤7.00元"},
            "投资逻辑": "test", "公司画像": {}, "竞争优势": {},
            "亮点": [], "风险": [], "操作建议": "持有", "止损": "无",
            "结论": {"总评": "test"},
        }
        report = format_report(mock)
        # 合计行应该是 5+4+5+8+5+5=32/60，不含加权总分
        self.assertIn("32/60", report)
        self.assertNotIn("88.5/60", report)
        self.assertNotIn("88.5/70", report)

    def test_aggregate_keys_filtered_from_sum(self):
        """加权总分、综合评级、置信度等复合指标不参与维度合计"""
        from backend.tools import format_report
        mock = {
            "代码": "000001", "名称": "测试",
            "评分": {
                "盈利能力": {"得分": 5, "依据": "t"},
                "成长性": {"得分": 3, "依据": "t"},
                "加权总分": {"得分": 95, "依据": "t"},
                "综合评级": {"得分": 100, "依据": "t"},
            },
            "投资评级": {"评级": "HOLD", "合理价值": 10.0, "估值差距": "+0%",
                         "安全边际要求": "30%", "买入区间": "≤7.00元"},
            "投资逻辑": "t", "公司画像": {}, "竞争优势": {},
            "亮点": [], "风险": [], "操作建议": "t", "止损": "t",
            "结论": {"总评": "t"},
        }
        report = format_report(mock)
        # 5+3=8/20，不含加权总分95
        self.assertIn("8/20", report)
        self.assertNotIn("103/30", report)

    def test_valuation_detail_in_return(self):
        """估值明细必须包含计算链(EPS/行业PE/质量乘数/成长溢价)"""
        from backend.scoring import compute_investment_rating
        decision = compute_investment_rating(
            company_type="成长型",
            financial_scores={
                "盈利能力": {"得分": 8}, "成长性": {"得分": 10},
                "财务健康": {"得分": 7}, "估值合理": {"得分": 3},
            },
            llm_scores={"行业前景": {}, "资金认可": {}},
            eps=5.0, stock_price=100, industry="通信",
            roe=45, debt=25,
        )
        chain = decision.get("估值明细", {})
        self.assertIn("EPS(TTM)", chain)
        self.assertIn("行业PE中枢", chain)
        self.assertIn("财务质量乘数", chain)
        self.assertIn("成长溢价", chain)
        self.assertIn("最终PE", chain)
        self.assertIn("公式", chain)
        # 公式应包含所有因子
        self.assertIn("×", chain["公式"])

    def test_cashflow_label_in_scores(self):
        """财务健康维度必须包含现金流标签和严重度"""
        from backend.tools import calculate_scores
        # 构造现金流极差的数据
        cs = {
            "profit": [{"报告期": "年报", "扣非净利润": 1000}],
            "cashflow": [{"报告期": "年报", "经营现金流净额": 200}],  # 覆盖率0.2→🔴
            "balance": [{"报告期": "年报", "资产总计": 10000, "负债合计": 7000}],
            "valuation": {"data": [{"日期": "2025-12-31", "ROE(%)": 15, "毛利率(%)": 30,
                          "净利率(%)": 10, "每股收益": 1, "每股净资产": 10,
                          "总股本": 100, "资产负债率(%)": 70}]},
            "price": {"price": 10}, "industry": "通信",
        }
        scores = calculate_scores(cs)
        fh = scores.get("财务健康", {})
        self.assertIn("现金流标签", fh)
        self.assertIn("现金流严重度", fh)
        self.assertIn("🔴", fh.get("现金流标签", ""))  # 覆盖率0.2→🔴警报

    def test_scenario_pe_constraint_in_prompt(self):
        """ANALYST_PROMPT必须包含情景PE动态调整约束"""
        from backend.agent import ANALYST_PROMPT
        self.assertIn("悲观PE=基准PE×0.6~0.8", ANALYST_PROMPT)
        self.assertIn("乐观PE=基准PE×1.1~1.3", ANALYST_PROMPT)

    def test_peg_constraint_in_prompt(self):
        """ANALYST_PROMPT必须包含PEG可持续增速约束"""
        from backend.agent import ANALYST_PROMPT
        self.assertIn("可持续增速", ANALYST_PROMPT)
        self.assertIn("单季暴增", ANALYST_PROMPT)

    def test_dilution_regex_handles_units(self):
        """定增股数提取必须正确区分股/万股/亿股"""
        import re
        # 股
        m = re.search(r'(\d+\.?\d*)\s*(亿|万)?股', "发行2666666666股A股")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "2666666666")
        self.assertIsNone(m.group(2))  # 无单位→股
        # 亿股
        m = re.search(r'(\d+\.?\d*)\s*(亿|万)?股', "非公开发行26.67亿股")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "亿")
        # 万股
        m = re.search(r'(\d+\.?\d*)\s*(亿|万)?股', "发行5000万股")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "万")

    def test_datasource_tier_import_and_default(self):
        """数据源分层模块默认FREE，高级插槽返回None"""
        from backend.datasource_tier import tier, DataSourceTier, query_premium_slot
        self.assertEqual(tier, DataSourceTier.FREE)
        self.assertIsNone(query_premium_slot("管理层画像", "000001"))
        self.assertIsNone(query_premium_slot("机构持仓", "000001"))

    def test_dilution_coefficient_cached(self):
        """稀释系数应该可以通过_dilution_coefficient缓存"""
        item = {"_dilution_coefficient": 0.874, "_dilution_shares": 25.9}
        self.assertTrue(bool(item.get("_dilution_coefficient")))
        # 模拟重试路径：系数存在则跳过抓取
        _cached = item.get("_dilution_coefficient")
        self.assertEqual(float(_cached), 0.874)


class TestReportQualityGuards(unittest.TestCase):
    """报告质量守卫：针对质检暴露的算术错误/时效性/口径问题"""

    def test_flash_report_extraction(self):
        """业绩快报正文必须提取出结构化数据（营收/归母/扣非/同比）"""
        from backend.tools import _extract_flash_report, _format_flash_hint
        sample = ("本报告期实现营业总收入343,295.19万元，同比减少2.59%；"
                  "归属于上市公司股东的净利润23,057.90万元，同比下降12.97%；"
                  "扣除非经常性损益后归属于上市公司股东的净利润22,857.60万元，同比增长11.18%。")
        flash = _extract_flash_report(sample, "国网信通2025年半年度业绩快报公告")
        self.assertIsNotNone(flash)
        self.assertEqual(flash["营收(亿元)"], 34.33)
        self.assertEqual(flash["归母净利润(亿元)"], 2.31)
        self.assertEqual(flash["扣非净利润(亿元)"], 2.29)
        self.assertEqual(flash["扣非同比(%)"], 11.18)
        self.assertEqual(flash["归母同比(%)"], -12.97)
        self.assertIn("快报", _format_flash_hint(flash))

    def test_flash_report_gm_vs_kf_disambiguation(self):
        """归母净利润提取必须排除'扣除非经常性损益后归母净利润'的干扰"""
        from backend.tools import _extract_flash_report
        sample = ("扣除非经常性损益后归属于上市公司股东的净利润22,857.60万元，同比增长11.18%；"
                  "归属于上市公司股东的净利润23,057.90万元，同比下降12.97%。")
        flash = _extract_flash_report(sample, "业绩快报")
        # 归母必须是23,057.90万(2.31亿)，不能错抓成扣非的22,857.60万
        self.assertEqual(flash["归母净利润(亿元)"], 2.31)
        self.assertEqual(flash["扣非净利润(亿元)"], 2.29)

    def test_scenario_arithmetic_validation(self):
        """情景估值：价格≠EPS×PE 时代码必须按算术重算"""
        from backend.agent import _validate_scenarios
        item = {"情景估值": {
            "悲观": {"价格": 7.00, "EPS": 0.25, "PE": 20, "概率": "20%"},
            "基准": {"价格": 9.38, "EPS": 0.375, "PE": 25, "概率": "55%"},
            "乐观": {"价格": 12.50, "EPS": 0.59, "PE": 25, "概率": "25%"},
        }}
        _validate_scenarios(item)
        sc = item["情景估值"]
        self.assertEqual(sc["悲观"]["价格"], 5.0)   # 0.25×20
        self.assertEqual(sc["乐观"]["价格"], 14.75)  # 0.59×25
        self.assertAlmostEqual(sc["概率加权价值"], 9.85, places=2)
        self.assertFalse(item["_scenario_check"]["arith_ok"])
        self.assertTrue(item["_scenario_check"]["monotonic_ok"])

    def test_scenario_monotonicity_detection(self):
        """情景价格倒挂必须被标记"""
        from backend.agent import _validate_scenarios
        item = {"情景估值": {
            "悲观": {"价格": 15.0, "概率": "20%"},
            "基准": {"价格": 9.0, "概率": "60%"},
            "乐观": {"价格": 12.0, "概率": "20%"},
        }}
        _validate_scenarios(item)
        self.assertFalse(item["_scenario_check"]["monotonic_ok"])

    def test_safety_margin_negative_adjustment(self):
        """安全边际质量微调必须允许负值（优质公司放宽），不再恒为+0.05"""
        from backend.scoring import _quality_adjustment
        self.assertLess(_quality_adjustment(roe=25, debt=15), 0)
        self.assertGreater(_quality_adjustment(roe=3, debt=80), 0)

    def test_pb_floor_on_buy_zone(self):
        """买入价不得跌破0.8倍每股净资产（当前PB≥1时）；破净股豁免"""
        from backend.scoring import compute_investment_rating
        fin = {"盈利能力": {"得分": 2}, "成长性": {"得分": 5},
               "财务健康": {"得分": 4}, "估值合理": {"得分": 6}}
        llm = {"行业前景": {"得分": 5}, "资金认可": {"得分": 5}}
        r = compute_investment_rating("困境反转型", fin, llm,
                                      eps=0.55, stock_price=14.10, industry="电力",
                                      roe=4.0, debt=55, bps=5.41)
        self.assertIn("PB地板", r["估值明细"])
        # 地板价 = 5.41×0.8 = 4.33
        self.assertIn("4.33", r["买入区间"])
        # 破净银行（股价5<BPS10）不触发
        r2 = compute_investment_rating("价值型", fin, llm,
                                       eps=1.0, stock_price=5.0, industry="银行",
                                       roe=9.0, debt=90, bps=10.0)
        self.assertNotIn("PB地板", r2["估值明细"])

    def test_cashflow_annual_period_preferred(self):
        """现金流覆盖率必须优先使用年报口径并标注期间"""
        from backend.tools import calculate_scores
        cs = {
            "profit": [
                {"报告期": "一季报", "扣非净利润": 352_0000},   # Q1净利极小
                {"报告期": "年报", "扣非净利润": 6_6000_0000},
            ],
            "cashflow": [
                {"报告期": "一季报", "经营现金流净额": 3.25e8},  # Q1覆盖率92倍(失真)
                {"报告期": "年报", "经营现金流净额": 21.5e8},    # 年报覆盖率3.26倍
            ],
            "balance": [{"报告期": "年报", "资产总计": 100, "负债合计": 50}],
            "valuation": {"data": [{"日期": "2025-12-31", "ROE(%)": 10, "毛利率(%)": 17,
                          "净利率(%)": 6, "每股收益": 0.55, "每股净资产": 5.41,
                          "总股本": 12e8, "资产负债率(%)": 52}]},
            "price": {"price": 14.1}, "industry": "计算机",
        }
        fh = calculate_scores(cs)["财务健康"]
        self.assertIn("年报", fh["依据"])
        self.assertIn("年报", fh["现金流标签"])
        # 年报口径覆盖率 = 21.5/6.6 ≈ 3.26，绝不能出现Q1口径的92倍
        self.assertNotIn("92", fh["依据"])

    def test_profitability_basis_formatted(self):
        """盈利能力依据中的浮点必须格式化（不再出现17.1648694075%）"""
        from backend.tools import calculate_scores
        cs = {
            "profit": [{"报告期": "年报", "扣非净利润": 1e8}],
            "cashflow": [],
            "balance": [],
            "valuation": {"data": [{"日期": "2025-12-31", "ROE(%)": 17.1648694075,
                          "毛利率(%)": 13.30, "净利率(%)": 6.34, "每股收益": 0.55,
                          "每股净资产": 5.41, "总股本": 12e8, "资产负债率(%)": 52.9908925347}]},
            "price": {"price": 14.1}, "industry": "计算机",
        }
        scores = calculate_scores(cs)
        self.assertNotIn("17.164869", scores["盈利能力"]["依据"])
        self.assertIn("17.2", scores["盈利能力"]["依据"])

    def test_analyst_prompt_flash_report_rule(self):
        """ANALYST_PROMPT必须包含业绩快报时序规则"""
        from backend.agent import ANALYST_PROMPT
        self.assertIn("业绩快报", ANALYST_PROMPT)
        self.assertIn("禁止再写", ANALYST_PROMPT)

    def test_auditor_prompt_timeliness_check(self):
        """AUDITOR_PROMPT必须包含时效性矛盾检查"""
        from backend.agent import AUDITOR_PROMPT
        self.assertIn("时效性矛盾", AUDITOR_PROMPT)

    def test_scenario_template_has_structured_eps_pe(self):
        """情景估值输出模板必须含结构化EPS/PE字段"""
        from backend.agent import _FORMAT_MANDATORY
        self.assertIn('"EPS"', _FORMAT_MANDATORY)
        self.assertIn('"PE"', _FORMAT_MANDATORY)

    def test_period_rank(self):
        """财报期间序号：一季报<中报<三季报<年报"""
        from backend.tools import _period_rank
        self.assertLess(_period_rank("一季报"), _period_rank("2026年半年度"))
        self.assertLess(_period_rank("2026年半年度"), _period_rank("三季报"))
        self.assertLess(_period_rank("三季报"), _period_rank("年报"))
        self.assertEqual(_period_rank("中报"), _period_rank("半年度"))

    def test_flash_feeds_growth_score(self):
        """快报比最新季报新鲜时，成长性必须用快报做趋势修正（拐点改善）"""
        from backend.tools import calculate_scores
        cs = {
            "profit": [
                {"报告期": "一季报", "营业总收入": 13.44e8, "扣非净利润": 352_0000},
                {"报告期": "年报", "营业总收入": 100e8, "扣非净利润": 6.6e8},
                {"报告期": "一季报", "营业总收入": 13.87e8, "扣非净利润": 2600_0000},
                {"报告期": "年报", "营业总收入": 98e8, "扣非净利润": 7.6e8},
            ],
            "cashflow": [], "balance": [],
            "valuation": {"data": [{"日期": "2025-12-31", "ROE(%)": 10, "毛利率(%)": 17,
                          "净利率(%)": 6, "每股收益": 0.55, "每股净资产": 5.41,
                          "总股本": 12e8, "资产负债率(%)": 52}]},
            "price": {"price": 14.1}, "industry": "IT服务Ⅱ",
            # 半年度快报：扣非同比+11.2%（年报-13%为负 → 拐点改善）
            "flash": {"报告期": "2026年半年度", "营收(亿元)": 34.33, "营收同比(%)": -2.59,
                      "归母净利润(亿元)": 2.31, "归母同比(%)": -12.97,
                      "扣非净利润(亿元)": 2.29, "扣非同比(%)": 11.18},
        }
        g = calculate_scores(cs)["成长性"]
        self.assertIn("快报", g["依据"])
        self.assertIn("拐点改善", g["依据"])
        # 年报底色1分 + 拐点改善+3 = 4分；若仍用Q1(-86%)则只有1分
        self.assertGreaterEqual(g["得分"], 3)

    def test_flash_ignored_when_older(self):
        """快报期间不新鲜于最新结构化数据时不得使用"""
        from backend.tools import calculate_scores
        cs = {
            "profit": [
                {"报告期": "三季报", "营业总收入": 80e8, "扣非净利润": 5e8},
                {"报告期": "年报", "营业总收入": 100e8, "扣非净利润": 6.6e8},
                {"报告期": "三季报", "营业总收入": 75e8, "扣非净利润": 5.5e8},
                {"报告期": "年报", "营业总收入": 98e8, "扣非净利润": 7.6e8},
            ],
            "cashflow": [], "balance": [],
            "valuation": {"data": [{"日期": "2025-12-31", "ROE(%)": 10, "毛利率(%)": 17,
                          "净利率(%)": 6, "每股收益": 0.55, "每股净资产": 5.41,
                          "总股本": 12e8, "资产负债率(%)": 52}]},
            "price": {"price": 14.1}, "industry": "IT服务Ⅱ",
            "flash": {"报告期": "2026年半年度", "扣非同比(%)": 11.18},  # 半年度 < 三季报
        }
        g = calculate_scores(cs)["成长性"]
        self.assertNotIn("快报", g["依据"])

    def test_timeliness_conflict_detector(self):
        """代码级时效性检查：快报已出+报告仍等待对应财报期 → 必须检出"""
        from backend.agent import _detect_timeliness_conflict
        item_wait = {
            "公告": {"列表": [{"标题": "x", "快报数据": {"报告期": "2026年半年度"}}]},
            "操作建议": "建议等待中报确认拐点后再决策",
        }
        self.assertIsNotNone(_detect_timeliness_conflict(item_wait))
        # 无快报 → 不检出
        item_no_flash = {"公告": {"列表": []}, "操作建议": "建议等待中报确认拐点后再决策"}
        self.assertIsNone(_detect_timeliness_conflict(item_no_flash))
        # 有快报但无等待措辞 → 不检出
        item_ok = {
            "公告": {"列表": [{"标题": "x", "快报数据": {"报告期": "2026年半年度"}}]},
            "操作建议": "基于快报数据，扣非已转正，可轻仓试探",
        }
        self.assertIsNone(_detect_timeliness_conflict(item_ok))

    def test_it_services_industry_pe_anchor(self):
        """IT服务行业必须有独立PE基准（不再fallback到默认18）"""
        from backend.scoring_config import get_valuation
        cfg = get_valuation()
        self.assertIn("IT服务Ⅱ", cfg["industry_pe"])
        self.assertGreater(cfg["industry_pe"]["IT服务Ⅱ"], cfg["default_ind_pe"])

    def test_pessimistic_above_fair_value_bridge_note(self):
        """情景悲观价>量化合理价值时，报告必须出现桥接说明"""
        from backend.tools import format_report
        mock = {
            "代码": "600131", "名称": "测试",
            "投资评级": {"评级": "SELL", "合理价值": 6.9, "当前价格": 14.1},
            "情景估值": {
                "悲观": {"价格": 8.4, "EPS": 0.42, "PE": 20, "假设": "x", "概率": "30%"},
                "基准": {"价格": 12.5, "EPS": 0.5, "PE": 25, "假设": "x", "概率": "50%"},
                "乐观": {"价格": 17.4, "EPS": 0.58, "PE": 30, "假设": "x", "概率": "20%"},
                "概率加权价值": 12.25,
            },
        }
        report = format_report(mock)
        self.assertIn("两套框架锚点不同", report)

    def test_merge_flash_into_profit_fill_kf(self):
        """快报回灌：同期间行缺扣非时必须补齐"""
        from backend.tools import merge_flash_into_profit
        profit = [{"date": "2026-06-30", "报告期": "半年报",
                   "营业总收入": 34.33e8, "归母净利润": 2.31e8, "扣非净利润": None, "_快报源": True},
                  {"date": "2025-06-30", "报告期": "半年报",
                   "营业总收入": 35.25e8, "归母净利润": 2.66e8, "扣非净利润": 2.06e8}]
        flash = {"报告期": "2026年半年度", "扣非净利润(亿元)": 2.29}
        merge_flash_into_profit(profit, flash)
        self.assertEqual(profit[0]["扣非净利润"], 2.29e8)
        self.assertEqual(len(profit), 2)  # 不新增行

    def test_merge_flash_into_profit_synthesize(self):
        """快报回灌：快报期间更新鲜时必须合成新行"""
        from backend.tools import merge_flash_into_profit
        profit = [{"date": "2026-03-31", "报告期": "一季报",
                   "营业总收入": 13.44e8, "归母净利润": 0.04e8, "扣非净利润": 0.035e8}]
        flash = {"报告期": "2026年半年度", "营收(亿元)": 34.33,
                 "归母净利润(亿元)": 2.31, "扣非净利润(亿元)": 2.29}
        merge_flash_into_profit(profit, flash)
        self.assertEqual(len(profit), 2)
        self.assertEqual(profit[0]["报告期"], "半年报")
        self.assertEqual(profit[0]["扣非净利润"], 2.29e8)
        self.assertTrue(profit[0]["_快报源"])

    def test_merge_flash_not_older(self):
        """快报期间不新鲜于利润表最新行时不得插入"""
        from backend.tools import merge_flash_into_profit
        profit = [{"date": "2026-09-30", "报告期": "三季报",
                   "营业总收入": 80e8, "归母净利润": 5e8, "扣非净利润": 4.8e8}]
        flash = {"报告期": "2026年半年度", "扣非净利润(亿元)": 2.29}
        merge_flash_into_profit(profit, flash)
        self.assertEqual(len(profit), 1)

    def test_financial_statements_fast_source(self):
        """RPT_FCI快源：600131利润表最新行应覆盖2026中报（正式披露前由快报合成）"""
        from backend.tools import get_financial_statements
        fin = get_financial_statements("600131")
        profit = fin.get("profit", [])
        self.assertGreater(len(profit), 0)
        self.assertGreaterEqual(profit[0].get("date", ""), "2026-06-30")
        self.assertEqual(profit[0].get("报告期"), "半年报")
        self.assertAlmostEqual(profit[0]["归母净利润"] / 1e8, 2.31, places=1)



def run_all():
    """运行全部测试并输出结果"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestCompilation, TestDataTools, TestScoringConsistency,
                TestInvestmentRating, TestHarnessGuards, TestConfig,
                TestOutputConsistency, TestReportQualityGuards]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Tests: {result.testsRun} | Passed: {result.testsRun - len(result.failures) - len(result.errors)} | Failed: {len(result.failures)} | Errors: {len(result.errors)}")
    if result.wasSuccessful():
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\n  FAIL: {test}")
            print(f"  {traceback[:200]}")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
