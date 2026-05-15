# Dummy Research Agent Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, Literal


class ResearchState(TypedDict):
    question: str
    search_history: list[str]
    gathered_info: list[str]
    next_query: str
    final_answer: str
    iteration_count: int


def search_tool(query: str) -> str:
    """Simulate search with fake database."""
    fake_db = {
        "langgraph": "LangGraph is a stateful workflow framework.",
        "agents": "Agents use loops and tools to solve tasks.",
        "memory": "Persistent memory enables long-term workflows."
    }
    return fake_db.get(query.lower(), "No results found.")


def agent_node(state: ResearchState) -> dict:
    """Decide next query or synthesize final answer."""
    iteration = state["iteration_count"]

    if iteration == 0:
        next_query = "langgraph"
    elif iteration == 1:
        next_query = "agents"
    else:
        return {
            "final_answer": (
                "LangGraph helps build stateful AI agents "
                "using loops and tools."
            )
        }

    return {"next_query": next_query}


def tool_node(state: ResearchState) -> dict:
    """Execute search and update knowledge."""
    query = state["next_query"]
    result = search_tool(query)

    return {
        "search_history": state["search_history"] + [query],
        "gathered_info": state["gathered_info"] + [result],
        "iteration_count": state["iteration_count"] + 1
    }


def should_continue(state: ResearchState) -> Literal["continue", "end"]:
    """Route based on whether final answer is ready."""
    if state.get("final_answer"):
        return "end"
    return "continue"


# Build graph
from langgraph.graph import StateGraph, START, END

builder = StateGraph(ResearchState)
builder.add_node("agent", agent_node)
builder.add_node("tool", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", should_continue, {"continue": "tool", "end": END}
)
builder.add_edge("tool", "agent")

graph = builder.compile()

# Run
result = graph.invoke(
    cast(ResearchState, {
        "question": "What is LangGraph?",
        "search_history": [],
        "gathered_info": [],
        "next_query": "",
        "final_answer": "",
        "iteration_count": 0
    })
)
print(f"[Question] {result['question']}")
print(f"\n[Searches] {result['search_history']}")
print(f"\n[Gathered Info]")
for info in result["gathered_info"]:
    print(f"  - {info}")
print(f"\n[Answer] {result['final_answer']}")
