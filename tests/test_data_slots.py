"""Tests for backend.data_slots and data slot fallback integration in backend.tools."""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.data_slots import (
    _read_data_slots,
    DataAPIFallbackChain,
    TushareProvider,
    query_data_slot,
    BaseDataProvider,
    _provider_for,
)


class TestReadDataSlots(unittest.TestCase):
    def test_empty_when_no_env(self):
        """未配置时返回空列表"""
        with patch.dict(os.environ, {}, clear=True):
            slots = _read_data_slots()
        self.assertEqual(slots, [])

    def test_reads_single_slot(self):
        """只配置 slot 1 时返回单个槽位"""
        env = {
            "DATA_SLOT_1_PROVIDER": "tushare",
            "DATA_SLOT_1_API_KEY": "abc123",
            "DATA_SLOT_1_BASE_URL": "",
            "DATA_SLOT_1_EXTRA": "foo=bar",
        }
        with patch.dict(os.environ, env, clear=True):
            slots = _read_data_slots()
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["provider"], "tushare")
        self.assertEqual(slots[0]["api_key"], "abc123")
        self.assertEqual(slots[0]["extra"], "foo=bar")

    def test_reads_multiple_slots(self):
        """配置 3 个槽位时按顺序返回"""
        env = {
            "DATA_SLOT_1_PROVIDER": "tushare",
            "DATA_SLOT_1_API_KEY": "k1",
            "DATA_SLOT_2_PROVIDER": "wind",
            "DATA_SLOT_2_API_KEY": "k2",
            "DATA_SLOT_3_PROVIDER": "ifind",
            "DATA_SLOT_3_API_KEY": "k3",
        }
        with patch.dict(os.environ, env, clear=True):
            slots = _read_data_slots()
        self.assertEqual(len(slots), 3)
        self.assertEqual([s["provider"] for s in slots], ["tushare", "wind", "ifind"])

    def test_ignores_unconfigured_slots(self):
        """中间槽位未配置时跳过"""
        env = {
            "DATA_SLOT_1_PROVIDER": "tushare",
            "DATA_SLOT_1_API_KEY": "k1",
            "DATA_SLOT_3_PROVIDER": "ifind",
            "DATA_SLOT_3_API_KEY": "k3",
        }
        with patch.dict(os.environ, env, clear=True):
            slots = _read_data_slots()
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[1]["provider"], "ifind")


class TestDataAPIFallbackChain(unittest.TestCase):
    def test_first_success_wins(self):
        """第一个 provider 成功时直接返回，不尝试后续"""
        p1 = MagicMock(spec=BaseDataProvider)
        p1.name = "p1"
        p1.fetch_stock_price.return_value = {"price": 10.0, "provider": "p1"}
        p2 = MagicMock(spec=BaseDataProvider)
        p2.name = "p2"

        chain = DataAPIFallbackChain([])
        chain._providers = [p1, p2]
        result = chain.invoke("stock_price", "300502")

        self.assertEqual(result["price"], 10.0)
        p2.fetch_stock_price.assert_not_called()

    def test_fallback_to_second_provider(self):
        """第一个失败时回退到第二个"""
        p1 = MagicMock(spec=BaseDataProvider)
        p1.name = "p1"
        p1.fetch_stock_price.return_value = {"error": "p1 failed", "provider": "p1"}
        p2 = MagicMock(spec=BaseDataProvider)
        p2.name = "p2"
        p2.fetch_stock_price.return_value = {"price": 20.0, "provider": "p2"}

        chain = DataAPIFallbackChain([])
        chain._providers = [p1, p2]
        result = chain.invoke("stock_price", "300502")

        self.assertEqual(result["price"], 20.0)
        self.assertIn("_data_slot_trace", result)
        self.assertEqual(result["_data_slot_trace"][0]["status"], "failed")
        self.assertEqual(result["_data_slot_trace"][1]["status"], "success")

    def test_all_failed_returns_error(self):
        """全部失败时返回错误和 trace"""
        p1 = MagicMock(spec=BaseDataProvider)
        p1.name = "p1"
        p1.fetch_stock_price.return_value = {"error": "p1 failed", "provider": "p1"}
        p2 = MagicMock(spec=BaseDataProvider)
        p2.name = "p2"
        p2.fetch_stock_price.return_value = {"error": "p2 failed", "provider": "p2"}

        chain = DataAPIFallbackChain([])
        chain._providers = [p1, p2]
        result = chain.invoke("stock_price", "300502")

        self.assertIn("error", result)
        self.assertIn("p2 failed", result["error"])
        self.assertEqual(len(result["_data_slot_trace"]), 2)

    def test_unimplemented_method_raises_attribute_error(self):
        """provider 未实现 category 方法时记录失败并继续"""
        p1 = MagicMock(spec=BaseDataProvider)
        p1.name = "p1"
        # 移除 fetch_stock_price 方法
        del p1.fetch_stock_price
        p2 = MagicMock(spec=BaseDataProvider)
        p2.name = "p2"
        p2.fetch_stock_price.return_value = {"price": 5.0}

        chain = DataAPIFallbackChain([])
        chain._providers = [p1, p2]
        result = chain.invoke("stock_price", "300502")
        self.assertEqual(result["price"], 5.0)


class TestProviderFactory(unittest.TestCase):
    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            _provider_for({"provider": "not_a_provider"})

    def test_tushare_provider_creation(self):
        p = _provider_for({"provider": "tushare", "api_key": "x", "base_url": "", "extra": ""})
        self.assertIsInstance(p, TushareProvider)


class TestTushareProviderMapping(unittest.TestCase):
    def test_ts_code_mapping(self):
        p = TushareProvider({"api_key": "", "base_url": "", "extra": ""})
        self.assertEqual(p._ts_code("600000"), "600000.SH")
        self.assertEqual(p._ts_code("000001"), "000001.SZ")
        self.assertEqual(p._ts_code("300502"), "300502.SZ")

    def test_fetch_financials_normalizes_columns(self):
        """Tushare 返回的英文列名应转换为 tools.py 内部中文列名"""
        import pandas as pd
        p = TushareProvider({"api_key": "", "base_url": "", "extra": ""})
        mock_client = MagicMock()

        income_df = pd.DataFrame([{
            "end_date": "20251231",
            "total_revenue": 1000.0,
            "oper_cost": 600.0,
            "n_income_attr_p": 120.0,
            "profit_dedt": 110.0,
            "sell_exp": 20.0,
            "admin_exp": 30.0,
            "fin_exp": 10.0,
            "operate_profit": 140.0,
            "income_tax": 20.0,
        }])
        balance_df = pd.DataFrame([{
            "end_date": "20251231",
            "total_assets": 2000.0,
            "total_liab": 800.0,
            "total_hldr_eqy_exc_min_int": 1200.0,
            "fix_assets": 300.0,
            "money_cap": 500.0,
            "accounts_receiv": 100.0,
            "inventories": 150.0,
            "acct_payable": 120.0,
        }])
        cashflow_df = pd.DataFrame([{
            "end_date": "20251231",
            "n_cashflow_act": 130.0,
            "n_cashflow_inv_act": -50.0,
            "n_cashflow_fin_act": -30.0,
            "c_paid_for_invest": 60.0,
            "depr_fa_coga_dpba": 20.0,
        }])

        mock_client.income.return_value = income_df
        mock_client.balancesheet.return_value = balance_df
        mock_client.cashflow.return_value = cashflow_df
        p._client = mock_client

        result = p.fetch_financials("300502")
        self.assertNotIn("error", result)
        self.assertEqual(len(result["profit"]), 1)
        self.assertEqual(result["profit"][0]["营业总收入"], 1000.0)
        self.assertEqual(result["profit"][0]["扣非净利润"], 110.0)
        self.assertEqual(result["balance"][0]["资产总计"], 2000.0)
        self.assertEqual(result["balance"][0]["资产负债率"], 40.0)
        self.assertEqual(result["cashflow"][0]["经营现金流净额"], 130.0)

    def test_fetch_financials_returns_error_on_api_failure(self):
        p = TushareProvider({"api_key": "bad", "base_url": "", "extra": ""})
        mock_client = MagicMock()
        mock_client.income.return_value = {"error": "api limit", "provider": "tushare"}
        p._client = mock_client
        result = p.fetch_financials("300502")
        self.assertIn("error", result)


class TestToolsFallbackIntegration(unittest.TestCase):
    def test_try_with_data_slots_when_primary_ok(self):
        """主结果有效时不应调用数据插槽"""
        from backend import tools
        primary = {"price": "100.0"}
        with patch("backend.data_slots.query_data_slot") as mock_query:
            result = tools._try_with_data_slots("300502", primary, "stock_price", ["price"])
        self.assertEqual(result, primary)
        mock_query.assert_not_called()

    def test_try_with_data_slots_when_primary_fails(self):
        """主结果失败时调用数据插槽"""
        from backend import tools
        primary = {"error": "sina failed"}
        fallback = {"price": "99.0", "provider": "tushare"}
        with patch("backend.data_slots.query_data_slot", return_value=fallback) as mock_query:
            result = tools._try_with_data_slots("300502", primary, "stock_price", ["price"])
        self.assertEqual(result["price"], "99.0")
        self.assertEqual(result["_fallback_from"], "data_slots")
        mock_query.assert_called_once_with("stock_price", "300502")

    def test_is_result_ok(self):
        from backend import tools
        self.assertTrue(tools._is_result_ok({"price": 10.0}, ["price"]))
        self.assertFalse(tools._is_result_ok({"error": "x"}, ["price"]))
        self.assertFalse(tools._is_result_ok({"price": None}, ["price"]))
        self.assertFalse(tools._is_result_ok({"price": ""}, ["price"]))
        self.assertFalse(tools._is_result_ok({"price": []}, ["price"]))


class TestQueryDataSlotEntry(unittest.TestCase):
    def test_no_slots_returns_error(self):
        with patch.dict(os.environ, {}, clear=True):
            result = query_data_slot("financials", "300502")
        self.assertIn("error", result)
        self.assertIn("未配置任何数据插槽", result["error"])


if __name__ == "__main__":
    unittest.main()
