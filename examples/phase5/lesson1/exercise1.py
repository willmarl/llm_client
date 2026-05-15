# StateGraph basics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast
from src import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class ChatState(TypedDict):
    user_input: str
    response: str


llm = get_llm()


def chatbot_node(state: ChatState) -> dict:
    """Call LLM and return response."""
    response = llm.invoke(state["user_input"])
    return {"response": response.content}


# Build graph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Run
result = graph.invoke(cast(ChatState, {"user_input": "Explain LangGraph simply"}))
print(result["response"])
