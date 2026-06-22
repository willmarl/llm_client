# Shared Memory Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, List
from llm_client import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class TeamState(TypedDict):
    user_request: str
    shared_memory: List[str]
    final_report: str
    completed: bool


llm = get_llm()


def supervisor_node(state: TeamState) -> dict:
    """Route agents based on how much shared memory has been collected."""
    if state.get("completed"):
        return {}

    if not state.get("shared_memory"):
        return {"current_agent": "researcher_a"}

    elif len(state["shared_memory"]) == 1:
        return {"current_agent": "researcher_b"}

    else:
        return {"current_agent": "writer"}


def route_supervisor(state: TeamState):
    if state.get("completed"):
        return END

    return state["current_agent"]


def researcher_a(state: TeamState) -> dict:
    result = "Python logging uses handlers and formatters."
    memory = state.get("shared_memory", [])
    memory.append(result)
    return {"shared_memory": memory}


def researcher_b(state: TeamState) -> dict:
    result = "Structured logging improves observability."
    memory = state.get("shared_memory", [])
    memory.append(result)
    return {"shared_memory": memory}


def writer_node(state: TeamState) -> dict:
    knowledge = "\n".join(state["shared_memory"])
    prompt = f"""
    Use this shared research:
    {knowledge}
    Write a final report.
    """
    response = llm.invoke(prompt)
    return {
        "final_report": response.content,
        "completed": True,
    }


# Build graph
builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher_a", researcher_a)
builder.add_node("researcher_b", researcher_b)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher_a": "researcher_a",
        "researcher_b": "researcher_b",
        "writer": "writer",
        END: END,
    },
)
builder.add_edge("researcher_a", "supervisor")
builder.add_edge("researcher_b", "supervisor")
builder.add_edge("writer", "supervisor")

graph = builder.compile()

# Run
result = graph.invoke(
    cast(
        TeamState,
        {
            "user_request": "Explain Python logging best practices",
            "shared_memory": [],
            "final_report": "",
            "completed": False,
        },
    )
)

print(f"[Request] {result['user_request']}")
print(f"\n[Shared Memory]")
for i, note in enumerate(result["shared_memory"], 1):
    print(f"  {i}. {note}")
print(f"\n[Final Report]\n{result['final_report']}")
