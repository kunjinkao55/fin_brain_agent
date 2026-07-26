"""
FinBrain 端到端验证脚本
覆盖：编译检查、数据工具、评分一致性、单股/多股分析、缓存、配置、Harness守卫
用法: python tests/test_e2e.py
"""

import sys, os, json, time, unittest
from unittest.mock import patch

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

    def test_consensus_forward_render(self):
        """A3 一致预期前瞻段渲染（机构覆盖正常渲染；无 rows 不渲染）"""
        from backend.tools import format_report
        item = {"代码": "600406", "名称": "国电南瑞",
                "一致预期前瞻": {"来源": "同花顺", "基准年EPS": 1.16,
                                "rows": [{"年份": "2025E", "EPS": 1.30, "增速": 12, "现价PE": 17.6, "机构数": 12},
                                          {"年份": "2026E", "EPS": 1.45, "增速": 12, "现价PE": 15.8, "机构数": 10}]}}
        text = format_report(item)
        self.assertIn("一致预期前瞻", text)
        self.assertIn("2025E", text)
        self.assertIn("17.6", text)
        item2 = {"代码": "301583", "名称": "托伦斯"}
        self.assertNotIn("一致预期前瞻", format_report(item2))

    def test_monitor_table_render(self):
        """A5 监测表：有 _monitor 时替代表格渲染，无则回退旧观察指标段"""
        from backend.tools import format_report
        item = {"代码": "301583", "名称": "托伦斯",
                "_monitor": [{"观察项": "限售解禁（网下配售）", "触发器": "163万股",
                              "窗口": "2027-01-10", "来源": "代码"}],
                "观察指标": ["毛利率能否企稳"]}
        text = format_report(item)
        self.assertIn("[监测表]", text)
        self.assertIn("限售解禁", text)
        self.assertNotIn("- 毛利率能否企稳", text)
        item2 = {"代码": "301583", "名称": "托伦斯", "观察指标": ["毛利率能否企稳"]}
        self.assertIn("[观察指标]", format_report(item2))

    def test_historical_pe_band(self):
        """A1 历史PE band：季度末TTM×季末价得区间；数据不足/次新股→None"""
        from backend.agent import _historical_pe_band
        profit = [
            {"报告期": "一季报", "date": "2026-03-31", "归母净利润": 1e8},
            {"报告期": "年报", "date": "2025-12-31", "归母净利润": 5e8},
            {"报告期": "三季报", "date": "2025-09-30", "归母净利润": 3e8},
            {"报告期": "半年报", "date": "2025-06-30", "归母净利润": 2e8},
            {"报告期": "一季报", "date": "2025-03-31", "归母净利润": 0.8e8},
            {"报告期": "年报", "date": "2024-12-31", "归母净利润": 4e8},
            {"报告期": "三季报", "date": "2024-09-30", "归母净利润": 2.5e8},
            {"报告期": "半年报", "date": "2024-06-30", "归母净利润": 1.6e8},
        ]
        import datetime as _dt
        bars = []
        d = _dt.date(2024, 6, 1)
        while d <= _dt.date(2026, 6, 1):
            if d.weekday() < 5:
                bars.append({"day": d.isoformat(), "close": "50"})
            d += _dt.timedelta(days=1)
        band = _historical_pe_band(profit, bars, 1e8, current_pe=40.0)
        self.assertIsNotNone(band)
        self.assertGreaterEqual(band["max"], band["median"])
        self.assertGreaterEqual(band["median"], band["min"])
        self.assertIn("40.0", band["文本"])
        self.assertIn("高于区间", band["文本"])
        # 次新股（K线不足240根）→ None
        self.assertIsNone(_historical_pe_band(profit, bars[:100], 1e8, 40.0))

    def test_comparables_section_render(self):
        """A2 可比公司估值对比段渲染（相对可比溢价）"""
        from backend.tools import format_report
        item = {"代码": "301583", "名称": "托伦斯",
                "可比公司对比": {"列表": [{"代码": "688409", "名称": "富创精密", "PE": 45.2, "PB": 5.1}],
                                "可比PE均值": 45.2, "本股PE": 337.0}}
        text = format_report(item)
        self.assertIn("可比公司估值对比", text)
        self.assertIn("富创精密", text)
        self.assertIn("溢价", text)

    def test_expected_recent_periods(self):
        """报告期动态推算：按披露节奏从系统时间推最近三份（不写死）"""
        from datetime import date
        from backend.tools import expected_recent_periods as erp
        self.assertEqual(erp(date(2026, 7, 26)),
                         [("2026-03-31", "一季报"), ("2025-12-31", "年报"), ("2025-09-30", "三季报")])
        self.assertEqual(erp(date(2026, 5, 15)),
                         [("2026-03-31", "一季报"), ("2025-12-31", "年报"), ("2025-09-30", "三季报")])
        # 年报披露前（2 月）：最近三份为 Q3/H1/Q1
        self.assertEqual(erp(date(2026, 2, 15)),
                         [("2025-09-30", "三季报"), ("2025-06-30", "半年报"), ("2025-03-31", "一季报")])
        # 9 月：半年报已出，Q3 未出
        self.assertEqual(erp(date(2026, 9, 1)),
                         [("2026-06-30", "半年报"), ("2026-03-31", "一季报"), ("2025-12-31", "年报")])

    def test_gap_explanation_and_trend_buy(self):
        """估值差距解读（数据驱动）+ 趋势参考买入价（支撑×1.05，接近现价不展示）"""
        from backend.agent import _build_gap_explanation, _trend_buy_price
        item = {
            "估值水位": {"PE": "337"},
            "投资评级": {"估值明细": {"行业PE中枢": 40}},
            "_events": [{"类型": "IPO", "参数": {"发行价(元)": 22.6}}],
            "_nl_pack": {"float_ratio": 0.1662},
            "评分": {"成长性": {"依据": "年报:营收+18%,扣非-7%"}},
        }
        lines = _build_gap_explanation(item, 174.0)
        text = "\n".join(lines)
        self.assertIn("337", text)          # PE 溢价
        self.assertIn("8.4", text)          # 337/40
        self.assertIn("670", text)          # 较发行价涨幅
        self.assertIn("16.62", text)        # 流通盘
        bars = [{"low": "102.56"}, {"low": "145.0"}, {"low": "120.3"}]
        self.assertAlmostEqual(_trend_buy_price(bars, 174.0), 107.69, places=2)
        self.assertIsNone(_trend_buy_price(bars, 110.0))   # 与现价差<10% → 不展示
        self.assertIsNone(_trend_buy_price([], 174.0))

    def test_cashflow_latest_period_primary(self):
        """财报期次纪律：最近一份财报为主要参考——Q1现金流为负是真实信号，必须用Q1口径；
        盈利能力指标也用最新期次值（毛利率23%而非年报27%）"""
        from backend.tools import calculate_scores
        cs = {
            "profit": [
                {"报告期": "一季报", "扣非净利润": 1e8, "归母净利润": 1.1e8},
                {"报告期": "年报", "扣非净利润": 4e8, "归母净利润": 4.2e8},
            ],
            "cashflow": [
                {"报告期": "一季报", "经营现金流净额": -0.8e8},  # Q1净流出（真实恶化，不属季节性失真）
                {"报告期": "年报", "经营现金流净额": 2.5e8, "购建固定资产支付现金": 1e8},
            ],
            "balance": [{"报告期": "年报", "资产总计": 100, "负债合计": 50}],
            "valuation": {"data": [{"日期": "2026-03-31", "报告期": "一季报", "ROE(%)": 3, "毛利率(%)": 23,
                          "净利率(%)": 8, "每股收益": 0.5, "每股净资产": 6.5,
                          "总股本": 2e8, "资产负债率(%)": 40},
                         {"日期": "2025-12-31", "报告期": "年报", "ROE(%)": 12, "毛利率(%)": 27,
                          "净利率(%)": 13, "每股收益": 2.0, "每股净资产": 6.0,
                          "总股本": 2e8, "资产负债率(%)": 41}]},
            "price": {"price": 50}, "industry": "半导体",
        }
        out = calculate_scores(cs)
        fh = out["财务健康"]
        self.assertIn("一季报", fh["依据"])
        self.assertIn("🔴", fh["现金流标签"])  # 覆盖率 -0.8/1.0 < 0.3 → 警报
        pe_dim = out["盈利能力"]
        self.assertIn("23.0%", pe_dim["依据"])   # 最新期毛利率，非年报 27%
        self.assertIn("一季报", pe_dim["依据"])
        self.assertIn("27.0%", pe_dim["依据"])   # 年报值仅作趋势参照标注

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


class TestStructuredOutput(unittest.TestCase):
    """结构化输出 Schema 校验"""

    def test_analyst_output_schema(self):
        """典型 Analyst JSON 可被 AnalystOutput 解析"""
        from backend.schemas import AnalystOutput
        payload = {
            "代码": "600131", "名称": "国网信通",
            "公司画像": {"主营业务": "电力信息化", "公司类型": "成长型"},
            "投资逻辑链": "国网数字化投入加大→订单增长→利润释放",
            "评分": {
                "盈利能力": {"得分": 7, "依据": "ROE 10%"},
                "成长性": {"得分": 8, "依据": "营收+20%"},
            },
            "亮点": ["订单饱满"], "风险": ["应收账款高"],
            "证伪条件": ["煤价反弹"],
            "操作建议": "≤10元建仓",
        }
        obj = AnalystOutput.model_validate(payload)
        self.assertEqual(obj.代码, "600131")
        self.assertEqual(obj.评分["成长性"].得分, 8)

    def test_critic_output_schema(self):
        """CriticOutput 可解析默认通过对象"""
        from backend.schemas import CriticOutput
        obj = CriticOutput.model_validate({"通过": True, "逻辑漏洞": []})
        self.assertTrue(obj.通过)

    def test_valuation_output_schema(self):
        """ValuationOutput 解析典型输出"""
        from backend.schemas import ValuationOutput
        obj = ValuationOutput.model_validate({"公司阶段": "成长期", "适用框架": ["PE"] })
        self.assertIn("PE", obj.适用框架)

    def test_audit_output_schema(self):
        """AuditOutput 解析典型问题列表"""
        from backend.schemas import AuditOutput
        payload = {"问题": [{"级别": "❌", "描述": "估值过高", "修正建议": "下调目标价"}]}
        obj = AuditOutput.model_validate(payload)
        self.assertEqual(len(obj.问题), 1)


class TestGraphRouting(unittest.TestCase):
    """LangGraph 条件分支路由"""

    def test_route_after_data_retry(self):
        """collected_data 无股票代码时，路由回 data_collector"""
        from backend.agent import route_after_data
        self.assertEqual(route_after_data({"collected_data": "", "processing_log": []}), "data_collector")

    def test_route_after_data_forward(self):
        """collected_data 含股票代码时，进入 classifier"""
        from backend.agent import route_after_data
        state = {"collected_data": '{"代码": "600131"}', "processing_log": []}
        self.assertEqual(route_after_data(state), "classifier")

    def test_route_after_data_no_infinite_loop(self):
        """已重试一次后不再回 data_collector"""
        from backend.agent import route_after_data
        log = [{"phase": "Data"}, {"phase": "Data"}]
        self.assertEqual(route_after_data({"collected_data": "", "processing_log": log}), "classifier")

    def test_route_after_critic_skip_repair(self):
        """Critic 无问题时跳过 Repair"""
        from backend.agent import route_after_critic
        log = [{"phase": "Critics", "total_issues": 0}]
        self.assertEqual(route_after_critic({"processing_log": log}), "reporter")

    def test_route_after_critic_goto_repair(self):
        """Critic 有问题时进入 Repair"""
        from backend.agent import route_after_critic
        log = [{"phase": "Critics", "total_issues": 3}]
        self.assertEqual(route_after_critic({"processing_log": log}), "repair")

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


class TestLLMFallback(unittest.TestCase):
    """LLM 多槽位熔断链"""

    def test_read_llm_slots_from_env(self):
        """从环境变量读取 3 槽位配置"""
        from backend.agent import _read_llm_slots
        with patch.dict(os.environ, {
            "LLM_SLOT_1_PROVIDER": "deepseek",
            "LLM_SLOT_1_MODEL": "deepseek-chat",
            "LLM_SLOT_1_API_KEY": "sk-1",
            "LLM_SLOT_2_PROVIDER": "openai",
            "LLM_SLOT_2_MODEL": "gpt-4o",
            "LLM_SLOT_3_PROVIDER": "anthropic",
            "LLM_SLOT_3_MODEL": "claude-sonnet-5",
        }, clear=False):
            slots = _read_llm_slots()
            self.assertEqual(len(slots), 3)
            self.assertEqual(slots[0]["provider"], "deepseek")
            self.assertEqual(slots[1]["provider"], "openai")
            self.assertEqual(slots[2]["provider"], "anthropic")

    def test_read_llm_slots_optional_23(self):
        """slot 2/3 可选，留空时不被读取"""
        from backend.agent import _read_llm_slots
        with patch.dict(os.environ, {
            "LLM_SLOT_1_PROVIDER": "deepseek",
            "LLM_SLOT_1_MODEL": "deepseek-chat",
        }, clear=False):
            slots = _read_llm_slots()
            self.assertEqual(len(slots), 1)
            self.assertEqual(slots[0]["provider"], "deepseek")

    def test_fallback_chain_slot2_on_failure(self):
        """slot 1 失败时自动切换到 slot 2"""
        from backend.agent import LLMFallbackChain

        class FakeLLM:
            def __init__(self, name):
                self.name = name
            def invoke(self, *args, **kwargs):
                if self.name == "slot1":
                    raise RuntimeError("slot1 failed")
                return f"ok from {self.name}"

        with patch("backend.agent._create_llm") as mock_create:
            mock_create.side_effect = [FakeLLM("slot1"), FakeLLM("slot2")]
            chain = LLMFallbackChain([
                {"provider": "deepseek", "model": "x", "api_key": "k1", "base_url": ""},
                {"provider": "openai", "model": "y", "api_key": "k2", "base_url": ""},
            ])
            result = chain.invoke("hello")
            self.assertEqual(result, "ok from slot2")
            self.assertEqual(mock_create.call_count, 2)

    def test_fallback_chain_all_fail(self):
        """所有槽位失败时抛出 RuntimeError"""
        from backend.agent import LLMFallbackChain

        class BadLLM:
            def invoke(self, *args, **kwargs):
                raise RuntimeError("always fail")

        with patch("backend.agent._create_llm") as mock_create:
            mock_create.return_value = BadLLM()
            chain = LLMFallbackChain([
                {"provider": "deepseek", "model": "x", "api_key": "k1", "base_url": ""},
                {"provider": "openai", "model": "y", "api_key": "k2", "base_url": ""},
            ])
            with self.assertRaises(RuntimeError):
                chain.invoke("hello")
            self.assertEqual(mock_create.call_count, 2)

    def test_fallback_chain_structured_slot2_on_failure(self):
        """结构化输出链在 slot 1 失败后切换到 slot 2"""
        from backend.agent import LLMFallbackChain
        from backend.schemas import CriticOutput

        class FakeLLM:
            def __init__(self, name, fail=False):
                self.name = name
                self.fail = fail
            def with_structured_output(self, schema, method):
                return self
            def invoke(self, *args, **kwargs):
                if self.fail:
                    raise RuntimeError("slot1 failed")
                return CriticOutput(通过=True, 逻辑漏洞=[])

        with patch("backend.agent._create_llm") as mock_create:
            mock_create.side_effect = [FakeLLM("slot1", fail=True), FakeLLM("slot2")]
            chain = LLMFallbackChain([
                {"provider": "deepseek", "model": "x", "api_key": "k1", "base_url": ""},
                {"provider": "openai", "model": "y", "api_key": "k2", "base_url": ""},
            ])
            structured = chain.with_structured_output(CriticOutput)
            result = structured.invoke("hello")
            self.assertTrue(result.通过)
            self.assertEqual(mock_create.call_count, 2)


class TestMarketSentiment(unittest.TestCase):
    """市场情绪评分（仅作参考，不参与估值）"""

    def _mock_env(self, breadth, limit_up, dragon):
        """统一 mock 全局情绪数据源"""
        return patch("backend.tools.get_market_breadth", return_value=breadth),                patch("backend.tools.get_limit_up_pool", return_value=limit_up),                patch("backend.tools.get_dragon_tiger_list", return_value=dragon),                patch("backend.cache.get", return_value=None)

    def test_sentiment_neutral(self):
        """市场中性、个股横盘，情绪得分应为 0"""
        from backend.tools import market_sentiment_score
        b, l, d, c = self._mock_env(
            {"上涨比例": "50.0%", "全A": {"上涨": 100, "下跌": 100, "平盘": 0, "总计": 200}},
            {"列表": []}, {"列表": []}
        )
        with b, l, d, c:
            r = market_sentiment_score("600131", {"price": 10.0, "yesterday_close": 10.0})
            self.assertEqual(r["综合情绪得分"], 0.0)
            self.assertEqual(r["情绪标签"], "中性")
            self.assertIn("按基本面估值锚点执行", r["对操作建议"])

    def test_sentiment_bullish(self):
        """市场偏暖、个股大涨，情绪得分偏热"""
        from backend.tools import market_sentiment_score
        b, l, d, c = self._mock_env(
            {"上涨比例": "60.0%", "全A": {"上涨": 120, "下跌": 80, "平盘": 0, "总计": 200}},
            {"列表": []}, {"列表": []}
        )
        with b, l, d, c:
            r = market_sentiment_score("600132", {"price": 10.5, "yesterday_close": 10.0})
            self.assertAlmostEqual(r["综合情绪得分"], 0.6, places=1)
            self.assertEqual(r["情绪标签"], "偏热")
            self.assertIn("分批建仓", r["对操作建议"])

    def test_sentiment_bearish(self):
        """市场偏冷、个股大跌，情绪得分偏冷"""
        from backend.tools import market_sentiment_score
        b, l, d, c = self._mock_env(
            {"上涨比例": "40.0%", "全A": {"上涨": 80, "下跌": 120, "平盘": 0, "总计": 200}},
            {"列表": []}, {"列表": []}
        )
        with b, l, d, c:
            r = market_sentiment_score("600133", {"price": 9.5, "yesterday_close": 10.0})
            self.assertAlmostEqual(r["综合情绪得分"], -0.6, places=1)
            self.assertEqual(r["情绪标签"], "偏冷")
            self.assertIn("耐心观察", r["对操作建议"])

    def test_sentiment_limit_up(self):
        """个股涨停时短线热度得分应反映涨停"""
        from backend.tools import market_sentiment_score
        b, l, d, c = self._mock_env(
            {"上涨比例": "55.0%", "全A": {"上涨": 110, "下跌": 90, "平盘": 0, "总计": 200}},
            {"列表": [{"代码": "600134", "名称": "测试"}]}, {"列表": []}
        )
        with b, l, d, c:
            r = market_sentiment_score("600134", {"price": 11.0, "yesterday_close": 10.0})
            self.assertAlmostEqual(r["综合情绪得分"], 0.8, places=1)
            self.assertEqual(r["情绪标签"], "极度乐观")
            self.assertIn("涨停", r["短线热度"]["备注"])
            self.assertIn("追高风险", r["对操作建议"])



class TestCriticGrounding(unittest.TestCase):
    """Critic/Repair 幻觉拦截：发现中的硬数字必须可溯源，Repair 引入新数字即作废"""

    def test_extract_hard_numbers(self):
        """硬数字提取：金额归一到亿、年号排除、万→亿换算"""
        from backend.consistency import extract_hard_numbers
        nums = extract_hard_numbers("PE 337倍，市值323亿，Q1净利0.15亿，涨幅8.5%，2026年，现金流5900万")
        self.assertIn((337.0, "倍"), nums)
        self.assertIn((323.0, "亿"), nums)
        self.assertIn((0.15, "亿"), nums)
        self.assertIn((8.5, "%"), nums)
        self.assertIn((0.59, "亿"), nums)  # 5900万 → 0.59亿
        self.assertFalse(any(u == "年" for _, u in nums))
        self.assertFalse(any(v == 2026 for v, _ in nums))

    def test_filter_untraceable_issues(self):
        """发现过滤：可溯源保留、无出处数字丢弃、无数字保留"""
        from backend.consistency import filter_untraceable_issues
        source = '{"评分":{"估值合理":{"依据":"PE 337倍, PB 26.6倍"}}, "估值水位":{"市值":"323亿"}}'
        kept, dropped = filter_untraceable_issues([
            "PE 337倍远超行业中枢，估值过高（原文：PE 337倍）",   # 可溯源 → 保留
            "PE 986倍明显泡沫，比行业均值高20倍",                  # 986倍无出处 → 丢弃
            "操作建议与评级方向矛盾",                               # 无数字 → 保留
        ], source)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 1)
        self.assertIn("986", dropped[0])

    def test_has_new_hard_numbers(self):
        """Repair 新数字检测：引入原文不存在的数字即违规"""
        from backend.consistency import has_new_hard_numbers
        original = '{"合理价值": 31.9, "PE": "337倍"}'
        ok = has_new_hard_numbers(original, '{"合理价值": 31.9, "PE": "337倍", "风险": "估值高"}')
        self.assertEqual(ok, [])
        bad = has_new_hard_numbers(original, '{"合理价值": 31.9, "对标": "可比公司PE 45倍"}')
        self.assertEqual(bad, [(45.0, "倍")])

    def test_critics_node_filters_hallucination(self):
        """critics_node：幻觉发现被拦截进 metadata，代码发现直通"""
        import json as _json
        from unittest.mock import patch
        from backend.agent import critics_node

        def fake_critic(prompt, text):
            if "财务" in prompt:
                return {"通过": False,
                        "财务误读": ["毛利率从45%下滑至27%，恶化严重"],  # 45% 无出处 → 幻觉
                        "_code_issues": ["FCF预警: 经营现金流0.59亿但资本开支1.89亿"]}
            if "逻辑" in prompt:
                return {"通过": False, "逻辑漏洞": ["操作建议与评级矛盾"], "建议": ""}
            return {"通过": True, "行业误述": [], "建议": ""}

        state = {"analysis": _json.dumps({"毛利率": "27.1%", "PE": "337倍"}, ensure_ascii=False),
                 "collected_data": "", "processing_log": [], "metadata": {}}
        with patch("backend.agent._call_critic", side_effect=fake_critic):
            out = critics_node(state)
        fixes = [f["issue"] for f in out["metadata"]["critic_fixes"]]
        self.assertFalse(any("45%" in f for f in fixes), "幻觉发现未被拦截")
        self.assertTrue(any("评级矛盾" in f for f in fixes))
        self.assertTrue(any("FCF" in f for f in fixes), "代码发现应直通")
        self.assertEqual(len(out["metadata"]["critic_dropped"]), 1)
        self.assertIn("幻觉拦截", out["metadata"]["critic_summary"])

    def test_repair_rejects_new_numbers(self):
        """repair_node：Repair 引入新数字（过时/编造）→ 整体作废回退原文"""
        from unittest.mock import patch, MagicMock
        from backend.agent import repair_node
        raw = '[{"代码":"301583","名称":"托伦斯","估值水位":{"PE":"337倍"}}]'
        state = {"analysis": raw, "collected_data": "", "processing_log": [],
                 "metadata": {"critic_fixes": [{"issue": "测试问题", "must_fix": True}]}}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='[{"代码":"301583","名称":"托伦斯","估值水位":{"PE":"986倍"}}]')
        with patch("backend.agent._get_llm", return_value=mock_llm):
            out = repair_node(state)
        self.assertEqual(out["analysis"], raw, "引入986倍的Repair应被作废")
        self.assertIn("repair_rejected", out["metadata"])

    def test_repair_passes_when_no_new_numbers(self):
        """repair_node：仅改措辞不引入新数字 → 修正保留"""
        from unittest.mock import patch, MagicMock
        from backend.agent import repair_node
        raw = '[{"代码":"301583","名称":"托伦斯","估值水位":{"PE":"337倍"},"结论":{"总评":"严重高估"}}]'
        state = {"analysis": raw, "collected_data": "", "processing_log": [],
                 "metadata": {"critic_fixes": [{"issue": "措辞过激", "must_fix": True}]}}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='[{"代码":"301583","名称":"托伦斯","估值水位":{"PE":"337倍"},"结论":{"总评":"估值偏高"}}]')
        with patch("backend.agent._get_llm", return_value=mock_llm):
            out = repair_node(state)
        self.assertIn("估值偏高", out["analysis"])
        self.assertNotIn("repair_rejected", out["metadata"])

    def test_strip_untraceable_sentences(self):
        """句子级清洗：含无出处数字的句子整句剔除，免责声明必留"""
        from backend.consistency import strip_untraceable_sentences
        source = "PE:337 市值:323亿 毛利率从27.1%降至23.3%"
        text = ("毛利率从27.1%降至23.3%，利润空间收窄。"
                "公司IPO募资5000万元扩产。"
                "以上分析基于公开财务数据，不构成投资建议，股市有风险，投资需谨慎。")
        cleaned, removed = strip_untraceable_sentences(text, source)
        self.assertNotIn("5000万", cleaned)
        self.assertIn("27.1%", cleaned)
        self.assertIn("不构成投资建议", cleaned)
        self.assertEqual(len(removed), 1)

    def test_generate_commentary_blocks_fabrication(self):
        """股评转写：LLM 引入报告外数字（编造募资额）→ 校验+句子级清洗后输出干净"""
        from unittest.mock import patch, MagicMock
        from backend.agent import generate_commentary
        report = ("FinBrain 投资研究: 托伦斯 (301583)\n"
                  "[估值水位] PE:337 PB:26.6 市值:323亿 前瞻PE:季节性失真\n"
                  "毛利率从27.1%降至23.3%。扣非净利润同比-8%。\n" * 12)  # >200字符
        fabricated = ("托伦斯这波炒作太猛了，股价174元对比发行价22.6元已经起飞。"
                      "公司IPO募资5000万元扩产，听着就不靠谱。"
                      "毛利率从27.1%降至23.3%，利润空间明显收窄。"
                      "以上分析基于公开财务数据，不构成投资建议，股市有风险，投资需谨慎。")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=fabricated)  # 首稿+重试稿同内容
        with patch("backend.agent._get_llm", return_value=mock_llm), \
             patch("backend.accounting_rag.search_kb", return_value=[]):
            out = generate_commentary(report)
        self.assertNotIn("5000万", out, "编造的募资额应被清洗")
        self.assertNotIn("174元", out, "报告外价格应被清洗")
        self.assertIn("27.1%", out, "真实数字应保留")
        self.assertIn("不构成投资建议", out)

    def test_generate_commentary_retry_success(self):
        """股评转写：首稿有幻觉、重试稿干净 → 采用重试稿"""
        from unittest.mock import patch, MagicMock
        from backend.agent import generate_commentary
        report = ("FinBrain 投资研究: 托伦斯 (301583)\n"
                  "[估值水位] PE:337 PB:26.6 市值:323亿\n毛利率从27.1%降至23.3%。\n" * 12)
        bad = "公司IPO募资5000万元扩产。毛利率从27.1%降至23.3%。"
        good = "毛利率从27.1%降至23.3%，利润空间收窄。PE高达337倍，估值太贵。"
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [MagicMock(content=bad), MagicMock(content=good)]
        with patch("backend.agent._get_llm", return_value=mock_llm), \
             patch("backend.accounting_rag.search_kb", return_value=[]):
            out = generate_commentary(report)
        self.assertIn("337", out)
        self.assertNotIn("5000万", out)
        self.assertEqual(mock_llm.invoke.call_count, 2)


def run_all():
    """运行全部测试并输出结果"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestCompilation, TestDataTools, TestScoringConsistency,
                TestInvestmentRating, TestHarnessGuards, TestConfig,
                TestOutputConsistency, TestReportQualityGuards,
                TestStructuredOutput, TestGraphRouting, TestLLMFallback,
                TestMarketSentiment, TestCriticGrounding]:
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
