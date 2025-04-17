from typing import Annotated
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import OllamaLLM
from langchain_huggingface import ChatHuggingFace
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    search_results: list

graph_builder = StateGraph(State)

# 配置 Ollama 模型，这里需要替换成你实际部署的模型名称
ollama_model = "deepseek-r1:1.5b"
try:
    llm = OllamaLLM(model=ollama_model, streaming=True)
    print(f"Successfully initialized OllamaLLM with model {ollama_model}")
except Exception as e:
    print(f"Failed to initialize OllamaLLM: {e}")
    raise

def filter_think_info(text):
    """过滤掉 <think> 和 </think> 及其包含的内容"""
    result = ""
    start_index = 0
    while True:
        think_start = text.find('<think>', start_index)
        if think_start == -1:
            result += text[start_index:]
            break
        result += text[start_index:think_start]
        think_end = text.find('</think>', think_start)
        if think_end == -1:
            break
        start_index = think_end + len('</think>')
    return result

def chatbot(state: State):
    show_thinking = False
    find_think_end = False
    input_text = " ".join([msg.content for msg in state["messages"]])
    print( "\n\n\n", "\n=============input_text=============\n" , input_text, "\n\n\n")
    print("Assistant: ", end = '')
    
    # 使用 stream 方法进行流式输出
    response_parts = []
    try:
        for chunk in llm.stream(input_text):
            if hasattr(chunk, 'content'):
                content = chunk.content
            elif isinstance(chunk, str):
                content = chunk
            else:
                continue
            
            if find_think_end is True or show_thinking is True:
                response_parts.append(content)
            else:
                start_idx = content.find('</think>')
                if start_idx == -1:
                    continue
                else:
                    find_think_end = True
                    content = content[start_idx + len('</think>') : len(content)]
            
            print(content, end="", flush=True)
            response_parts.append(content)

    except Exception as e:
        print(f"Error during streaming: {e}")
        return {"messages": [{"role": "assistant", "content": "An error occurred during streaming."}]}
    print("\n\n")
    
    response = ''.join(response_parts)

    # 添加 role 键
    return {"messages": [{"role": "assistant", "content": response}]}

def run_search(state: State):
    input_text = " ".join([msg.content for msg in state["messages"]])
    results = tool.run(input_text)
    print( "\n\n\n", "\n=============search result-=============\n" , results, "\n\n\n")
    return {"search_results": results}

# Add node.
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("search", run_search)
# 添加聊天机器人节点
graph_builder.add_node("chatbot", chatbot)
# 设置图的入口点为搜索节点
graph_builder.set_entry_point("search")
# 设置图的结束点为聊天机器人节点
graph_builder.add_edge("search", "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    try:
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values"):
            pass  # 由于在 chatbot 函数中已经处理了流式输出，这里不需要再处理
    except Exception as e:
        print(f"Error in stream_graph_updates: {e}")

while True:
    try:
        user_input = input("User: \n\n")
        print("\n")
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