# -*- coding: utf-8 -*-
"""离线回归测试：上下文必须按轮次压缩、保留相关旧内容并受预算约束。"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local")

# 部分精简开发环境未安装 langgraph-checkpoint-sqlite。上下文函数不依赖它，
# 用最小桩件避免持久化适配器阻断离线单元测试。
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
except ModuleNotFoundError:
    sqlite_module = types.ModuleType("langgraph.checkpoint.sqlite")
    sqlite_module.SqliteSaver = type("SqliteSaver", (), {})
    sys.modules["langgraph.checkpoint.sqlite"] = sqlite_module


class TestContextManagement(unittest.TestCase):
    def setUp(self):
        from backend import agent
        self.agent = agent
        self.original = (agent.CONTEXT_RECENT_TURNS, agent.CONTEXT_TRIGGER_CHARS,
                         agent.CONTEXT_MAX_CHARS, agent.CONTEXT_MEMORY_CHARS,
                         agent.CONTEXT_STRUCTURED_FACTS)
        agent.CONTEXT_RECENT_TURNS = 2
        agent.CONTEXT_TRIGGER_CHARS = 100
        agent.CONTEXT_MAX_CHARS = 1_500
        agent.CONTEXT_MEMORY_CHARS = 400
        agent.CONTEXT_STRUCTURED_FACTS = 24

    def tearDown(self):
        (self.agent.CONTEXT_RECENT_TURNS, self.agent.CONTEXT_TRIGGER_CHARS,
         self.agent.CONTEXT_MAX_CHARS, self.agent.CONTEXT_MEMORY_CHARS,
         self.agent.CONTEXT_STRUCTURED_FACTS) = self.original

    def test_keeps_complete_recent_turns_and_system_memory(self):
        history = [
            {"role": "user", "content": "分析贵州茅台 600519 的估值"},
            {"role": "assistant", "content": "结论：需要结合 PE 和现金流判断。"},
            {"role": "user", "content": "再看宁德时代 300750"},
            {"role": "assistant", "content": "风险：行业价格竞争。"},
            {"role": "user", "content": "继续解释仓位"},
            {"role": "assistant", "content": "仓位应和风险承受能力匹配。"},
            {"role": "user", "content": "最新问题"},
            {"role": "assistant", "content": "最新回答"},
        ]
        result = self.agent.compress_history(history, query="600519 现在的估值")
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("长期对话记忆", result[0]["content"])
        self.assertEqual(result[-4:], history[-4:])
        self.assertTrue(any(m["content"] == "分析贵州茅台 600519 的估值" for m in result))

    def test_structured_memory_keeps_category_entity_and_source(self):
        history = [
            {"role": "user", "content": "分析贵州茅台 600519，并关注仓位"},
            {"role": "assistant", "content": "风险：估值偏高，仓位不超过 20%。"},
            {"role": "user", "content": "继续跟踪业绩公告"},
            {"role": "assistant", "content": "公告：下次财报披露前需要复核净利润同比。"},
        ]
        memory = self.agent.get_structured_memory(history)
        self.assertIn("600519", memory["entities"])
        risk = next(fact for fact in memory["facts"] if "估值偏高" in fact["content"])
        self.assertEqual(risk["category"], "风险")
        self.assertEqual(risk["source_turn"], 1)
        self.assertIn("600519", risk["symbols"])

    def test_respects_budget_and_never_calls_llm(self):
        history = []
        for index in range(8):
            history.extend([
                {"role": "user", "content": f"问题 {index} 00000{index} " + "甲" * 500},
                {"role": "assistant", "content": "结论 " + "乙" * 500},
            ])
        result = self.agent.compress_history(history, query="000007")
        self.assertLessEqual(sum(len(m["content"]) for m in result), self.agent.CONTEXT_MAX_CHARS)
        self.assertEqual(result[-1]["role"], "assistant")

    def test_short_history_is_unchanged(self):
        history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}]
        self.assertEqual(self.agent.compress_history(history, query="你好"), history)


if __name__ == "__main__":
    unittest.main()
