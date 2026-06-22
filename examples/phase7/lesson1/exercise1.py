# Thread ID / Memory Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, Annotated, cast
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from rich import print


class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]


def chatbot_node(state: ChatState) -> dict:
    """Respond to last message, with access to full history."""
    messages = state["messages"]
    last = messages[-1]

    if "my name" in last.lower():
        for msg in messages[:-1]:
            if "my name is" in msg.lower():
                name = msg.lower().split("my name is")[-1].strip().split()[0].capitalize()
                return {"messages": [f"Your name is {name}!"]}

    return {"messages": [f"AI Response to: {last}"]}


# Build graph
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = cast(RunnableConfig, {"configurable": {"thread_id": "session-1"}})

# Run 1: introduce name
result = graph.invoke({"messages": ["My name is Alex"]}, config=config)
print(f"[session-1 | run 1] {result['messages']}")

# Run 2: ask about name — same thread, history is preserved
result = graph.invoke({"messages": ["What is my name?"]}, config=config)
print(f"[session-1 | run 2] {result['messages']}")

# Different thread: no memory of Alex
config2 = cast(RunnableConfig, {"configurable": {"thread_id": "session-2"}})
result = graph.invoke({"messages": ["What is my name?"]}, config=config2)
print(f"\n[session-2 | no memory] {result['messages']}")
