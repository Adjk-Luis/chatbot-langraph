from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langgraph.types import Command, interrupt
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END

# vLLM based llm.
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()
# llm.bind_tools()
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    print("========== human_assistance called ================")
    try:
        # human_response = interrupt(
        #     {
        #         "question": "Is this correct?",
        #         "name": name,
        #         "birthday": birthday,
        #     },
        # )
        question = f"Is this correct? Name: {name}, Birthday: {birthday}. (yes/no): "
        human_response = input(question).strip().lower()

    except Exception as e:
        print(f"An error occurred in interrupt function: {e}")
        
        # 可以根据具体情况返回一个错误信息给调用者
        error_response = f"Error in interrupt: {e}"
        state_update = {
            "name": name,
            "birthday": birthday,
            "messages": [ToolMessage(error_response, tool_call_id=tool_call_id)],
        }
        return Command(update=state_update)

    if human_response == "yes" or human_response == "y":
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = input("Please input correct name: ")
        verified_birthday = input("Please input correct birthday: ")
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]

# 使用 ChatOllama 并指定本地部署的模型名称
llm = ChatOllama(model="qwen2.5:latest")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input: str):
    events = graph.stream({"messages": [{"role": "user", "content": user_input}]},
                          {"configurable": {"thread_id": "2"}},
                          stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)

try:
    img_data = graph.get_graph().draw_png()
    with open('graph.png', 'wb') as f:
        f.write(img_data)
    print("Graph saved as graph.png. You can open it manually.")
except Exception:
    print("This requires some extra dependencies and is optional")
    pass