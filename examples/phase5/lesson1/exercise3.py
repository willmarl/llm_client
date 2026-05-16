# Logger Node Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast
from llm_client import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class ChatState(TypedDict):
    user_input: str
    response: str


llm = get_llm()


def logger_node(state: ChatState) -> dict:
    """Log the current state."""
    print("[LOG] State:")
    print(f"  user_input: {state['user_input']}")
    if "response" in state and state["response"]:
        print(f"  response: {state['response'][:100]}...")
    return {}


def chatbot_node(state: ChatState) -> dict:
    """Call LLM and return response."""
    response = llm.invoke(state["user_input"])
    return {"response": response.content}


# Build graph with logger
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("logger", logger_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "logger")
graph_builder.add_edge("logger", END)

graph = graph_builder.compile()

# Run
result = graph.invoke(cast(ChatState, {"user_input": "What is AI?"}))
print("\n[FINAL]")
print(result["response"])
