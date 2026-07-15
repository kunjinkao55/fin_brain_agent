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


def run_all():
    """运行全部测试并输出结果"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestCompilation, TestDataTools, TestScoringConsistency,
                TestInvestmentRating, TestHarnessGuards, TestConfig]:
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
