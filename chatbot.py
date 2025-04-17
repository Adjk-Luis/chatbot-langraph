from typing import Annotated
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import OllamaLLM
from IPython.display import Image, display

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 配置 Ollama 模型，这里需要替换成你实际部署的模型名称
ollama_model = "deepseek-r1:1.5b"
try:
    llm = OllamaLLM(model=ollama_model, streaming=True)
    print(f"Successfully initialized OllamaLLM with model {ollama_model}")
except Exception as e:
    print(f"Failed to initialize OllamaLLM: {e}")
    raise

def chatbot(state: State):
    input_text = " ".join([msg.content for msg in state["messages"]])
    # 使用 stream 方法进行流式输出
    response_parts = []
    try:
        print("Starting streaming...")  # 添加调试信息
        for chunk in llm.stream(input_text):
            # 修改此处，直接处理字符串
            if isinstance(chunk, str):
                response_parts.append(chunk)
                print(chunk, end="", flush=True)
            else:
                response_parts.append(chunk.content)
                print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"Error during streaming: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred during streaming."}]}
    response = ''.join(response_parts)
    print()  # 换行

    # 添加 role 键
    return {"messages": [{"role": "assistant", "content": response}]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values"):
            pass  # 由于在 chatbot 函数中已经处理了流式输出，这里不需要再处理
    except Exception as e:
        print(f"Error in stream_graph_updates: {e}")

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"Unexpected error: {e}")
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


try:
    img_data = graph.get_graph().draw_mermaid_png()
    with open('graph.png', 'wb') as f:
        f.write(img_data)
    print("Graph saved as graph.png. You can open it manually.")
except Exception:
    print("This requires some extra dependencies and is optional")
    pass