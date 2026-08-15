"""
FinBrain 评估套件 — 审查链 fault-injection 诊断。

目的（开发期调试定位）：不改真实流水线、不调 LLM、不访问网络，
把"已知错误类型"注入到黄金数据中，逐条验证审查链的确定性守卫能否检出，
产出每类故障的检出率报告 —— 哪条守卫回归一跑即知。

目录：
    faults.py   故障目录（黄金数据 + 注入 + 检测器）       —— Layer C 审查链
    harness.py  run_fault_injection() 诊断运行器 + 报告聚合
    analyst.py  Layer B 推理链组件级评估（analyst 节点指标）
    data.py     Layer A 数据管道黄金比对（data_collector 抓取正确性）
"""

from backend.eval_suite.harness import run_fault_injection, format_report
from backend.eval_suite.analyst import (
    evaluate_analysis, evaluate_analyst, format_analyst_report,
    GOLDEN_COLLECTED, GOLDEN_FACTS, REQUIRED_FIELDS,
)
from backend.eval_suite.data import (
    extract_metrics, check_against_golden,
    evaluate_data_pipeline, format_pipeline_report, collect_data,
    GOLDEN_FACTS as DATA_GOLDEN_FACTS,
)
from backend.eval_suite.golden_pool import (
    GOLDEN_STOCKS, SECTORS, get_stock,
    get_real_symbols, build_golden_collected, build_golden_facts,
    build_golden_data_facts,
)

__all__ = [
    "run_fault_injection", "format_report",
    "evaluate_analysis", "evaluate_analyst", "format_analyst_report",
    "GOLDEN_COLLECTED", "GOLDEN_FACTS", "REQUIRED_FIELDS",
    "extract_metrics", "check_against_golden",
    "evaluate_data_pipeline", "format_pipeline_report", "collect_data",
    "DATA_GOLDEN_FACTS",
    "GOLDEN_STOCKS", "SECTORS", "get_stock",
    "get_real_symbols", "build_golden_collected", "build_golden_facts",
    "build_golden_data_facts",
]
