"""FinBrain 启动脚本 — 从项目根目录运行: python run.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from backend.agent import build_graph, _get_chat_agent, _get_phantom_agent, \
    _classify_request, compress_history, _dicts_to_messages
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("FinBrain Agent")
    print("Type 'quit' to exit, 'clear' to reset context")
    print()

    graph = build_graph()
    history = []

    while True:
        try:
            user_input = input("\n>$ ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("bye"); break
            if user_input.lower() == "clear":
                history = []; print("[context cleared]"); continue
            if not user_input: continue

            history = compress_history(history)
            req_type = _classify_request(user_input)

            if req_type == "phantom":
                phantom = _get_phantom_agent()
                msgs = _dicts_to_messages(history)
                msgs.append(HumanMessage(content=user_input))
                reply = phantom.invoke({"messages": msgs})["messages"][-1].content
            elif req_type == "analysis":
                lc_messages = _dicts_to_messages(history)
                lc_messages.append(HumanMessage(content=user_input))
                result = graph.invoke({
                    "messages": lc_messages, "user_question": user_input,
                    "collected_data": "", "analysis": "", "report": "",
                    "processing_log": [], "sentiment_map": {},
                })
                reply = result.get("report") or result["messages"][-1].content
            else:
                chat = _get_chat_agent()
                msgs = _dicts_to_messages(history)
                msgs.append(HumanMessage(content=user_input))
                reply = chat.invoke({"messages": msgs})["messages"][-1].content

            print(reply)
            print("-" * 60)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nbye"); break
        except Exception as e:
            print(f"[Error] {e}")
