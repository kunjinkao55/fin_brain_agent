# -*- coding: utf-8 -*-
"""Architecture refactor test: _fix_and_decide is split into four phases.

验证：
- backend/agent.py 中 reporter_node 内部已定义 _prepare_data / _calculate_valuation / _apply_guards / _finalize_decision
- _fix_and_decide 仅作为四阶段编排器
- 审计重试调用 _apply_guards + _finalize_decision，而非完整 _fix_and_decide
- reporter_node 在 mock 数据下可正常输出报告

全部为 mock 数据，不访问网络/LLM。
"""
import os
import sys
import re
import ast
import json
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")


def _mock_fin(sym):
    return {
        "profit": [
            {"报告期": "年报", "date": "2024-12-31", "归母净利润": 1_000_000_000.0,
             "营业总收入": 5_000_000_000.0},
            {"报告期": "年报", "date": "2023-12-31", "归母净利润": 800_000_000.0,
             "营业总收入": 4_000_000_000.0},
        ],
        "cashflow": [
            {"报告期": "年报", "经营现金流净额": 1_200_000_000.0,
             "购建固定资产支付现金": 300_000_000.0},
        ],
        "balance": [
            {"报告期": "年报", "股东权益": 8_000_000_000.0, "货币资金": 1_000_000_000.0,
             "短期借款": 500_000_000.0, "长期借款": 200_000_000.0},
        ],
    }


def _mock_val(sym):
    return {
        "data": [
            {"报告期": "年报", "date": "2024-12-31", "每股收益": 1.0,
             "每股净资产": 8.0, "ROE(%)": 12.5, "资产负债率(%)": 30.0,
             "总股本": 1_000_000_000.0},
        ]
    }


def _mock_price(sym):
    return {"price": 20.0}


def _mock_ind(sym):
    return {"行业": "电子元件", "industry_name": "电子元件"}


def _mock_ann(sym, count=20):
    return {"列表": []}


class TestArchitectureDefinitions(unittest.TestCase):
    """静态检查：四个阶段函数与编排器是否存在于源码中。"""

    def test_four_phases_defined(self):
        """reporter_node 函数体内必须定义四个阶段函数。"""
        with open("backend/agent.py", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)

        reporter_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "reporter_node":
                reporter_node = node
                break
        self.assertIsNotNone(reporter_node, "reporter_node 未找到")

        inner_names = {
            n.name for n in reporter_node.body
            if isinstance(n, ast.FunctionDef)
        }
        required = {"_prepare_data", "_calculate_valuation", "_apply_guards", "_finalize_decision"}
        missing = required - inner_names
        self.assertFalse(missing, f"reporter_node 内缺少阶段函数: {missing}")

    def test_fix_and_decide_is_orchestrator(self):
        """_fix_and_decide 仅作为四阶段编排器，不再包含大段实现代码。"""
        with open("backend/agent.py", encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)

        reporter_node = next(n for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef) and n.name == "reporter_node")
        fix_fn = next((n for n in reporter_node.body
                       if isinstance(n, ast.FunctionDef) and n.name == "_fix_and_decide"), None)
        self.assertIsNotNone(fix_fn, "_fix_and_decide 未找到")
        # 编排器应调用四个阶段 + 兜底修复（当前价格/安全边际），代码行数不超过 40 行
        self.assertLessEqual(fix_fn.end_lineno - fix_fn.lineno, 40,
                             "_fix_and_decide 看起来仍是巨型函数")

    def test_audit_retry_uses_phased_calls(self):
        """审计重试应调用 _apply_guards + _finalize_decision，而非 _fix_and_decide。"""
        with open("backend/agent.py", encoding="utf-8") as f:
            src = f.read()
        # 简单文本断言：重试区域出现 _apply_guards 且与 _finalize_decision 相邻
        self.assertRegex(src, r"_apply_guards\(item,\s*sym\)\s+_finalize_decision\(item,\s*sym\)")


class TestArchitectureRuntime(unittest.TestCase):
    """运行时检查：reporter_node 在 mock 数据下能完成四个阶段并输出报告。"""

    @patch("backend.tools.get_financial_statements", side_effect=_mock_fin)
    @patch("backend.tools.get_valuation", side_effect=_mock_val)
    @patch("backend.tools.fetch_stock_price", side_effect=_mock_price)
    @patch("backend.tools.get_industry_info", side_effect=_mock_ind)
    @patch("backend.tools.get_recent_announcements", side_effect=_mock_ann)
    def test_reporter_node_runs_without_error(self, _ann, _ind, _price, _val, _fin):
        from backend.agent import reporter_node

        analysis = json.dumps({
            "代码": "000001",
            "名称": "测试银行",
            "评分": {
                "盈利能力": {"得分": 7, "依据": "ROE 12.5%"},
                "成长性": {"得分": 6, "依据": "营收增速25%"},
                "财务健康": {"得分": 8, "依据": "负债率30%"},
                "估值合理": {"得分": 6, "依据": "PE 20 倍"},
            },
            "公司画像": {"公司类型": "价值型"},
            "投资逻辑": ["测试"],
            "风险": ["测试风险"],
            "操作建议": "测试建议",
            "结论": {"综合判断": "测试"},
        }, ensure_ascii=False)

        state = {
            "analysis": analysis,
            "collected_data": "",
            "metadata": {},
            "processing_log": [],
        }

        result = reporter_node(state)
        self.assertIsInstance(result, dict)
        self.assertIn("report", result)
        self.assertIsInstance(result["report"], str)
        self.assertGreater(len(result["report"]), 100)


if __name__ == "__main__":
    unittest.main()
