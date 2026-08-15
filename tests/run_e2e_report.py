# -*- coding: utf-8 -*-
"""真实端到端：分析托伦斯 301583，验证完整报告生成（修复 None 崩溃后）。"""
import os, sys, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FINBRAIN_DATA_MODE", "local")
os.environ.setdefault("FINBRAIN_LLM_MODE", "local_client")

from backend.agent import build_graph

USER_QUESTION = "分析托伦斯301583"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "e2e_report_301583.txt")

def main():
    t0 = time.time()
    graph = build_graph()
    result = graph.invoke(
        {
            "messages": [],
            "user_question": USER_QUESTION,
            "collected_data": "",
            "analysis": "",
            "report": "",
            "processing_log": [],
            "sentiment_map": {},
        },
        {"configurable": {"thread_id": f"e2e_{int(time.time()*1000)}"}},
    )
    elapsed = time.time() - t0
    report = result.get("report") or ""
    analysis = result.get("analysis") or ""

    print(f"耗时 {elapsed:.1f}s")
    print(f"report 长度: {len(report)}")
    print(f"analysis 长度: {len(analysis)}")

    # 检查是否出现错误
    if "[Report Error]" in report:
        print("!! 报告仍含 [Report Error]")
        for line in report.splitlines():
            if "Report Error" in line:
                print("   ", line)
    else:
        print("OK: 报告无 [Report Error]")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {OUT}")

    # 打印报告头部
    print("\n===== 报告预览 =====")
    print(report[:1500])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FAILED:", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)
