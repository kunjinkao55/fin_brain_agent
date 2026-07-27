"""真实端到端测试：分析京东方A（000725），验证 INV-11 叙事真值校验与预告解析。"""
import os, sys, time, traceback
sys.path.insert(0, os.path.dirname(__file__))

from backend.agent import build_graph
from langchain_core.messages import HumanMessage

OUT_PATH = os.path.join(os.path.dirname(__file__), "example", "test_report_000725_boe.txt")
USER_QUESTION = "分析京东方A"


def main():
    start = time.time()
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=USER_QUESTION)],
            "user_question": USER_QUESTION,
            "collected_data": "",
            "analysis": "",
            "report": "",
            "processing_log": [],
            "sentiment_map": {},
        },
        {"configurable": {"thread_id": "boe_test"}},
    )
    report = result.get("report") or result["messages"][-1].content
    elapsed = time.time() - start

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"# 京东方A（000725）测试报告\n")
        f.write(f"生成耗时: {elapsed:.1f}s\n\n")
        f.write(report)

    print(f"[OK] 报告已保存至 {OUT_PATH}")
    print(f"[INFO] 耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write(f"[Error] 测试失败: {e}\n\n{traceback.format_exc()}")
        print(f"[Error] 测试失败，详情见 {OUT_PATH}")
        raise
