# Reflection System Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, Literal
from langgraph.graph import StateGraph, START, END
from rich import print


class ReflectionState(TypedDict):
    user_request: str
    draft: str
    critique: str
    improved_draft: str
    quality_score: int
    iteration_count: int
    final_answer: str


def generate_node(state: ReflectionState) -> dict:
    """Generate initial draft."""
    draft = """def add(a,b):
return a+b"""
    return {"draft": draft}


def critique_node(state: ReflectionState) -> dict:
    """Critique the draft and assign quality score."""
    draft = state["draft"]
    critique = []

    if "return" in draft and "    return" not in draft:
        critique.append("Indentation is incorrect.")

    if "def add(a,b)" in draft:
        critique.append("Spacing should follow PEP8.")

    if critique:
        score = 4
    else:
        score = 9

    return {
        "critique": "\n".join(critique),
        "quality_score": score
    }


def improve_node(state: ReflectionState) -> dict:
    """Improve the draft based on critique."""
    improved = """def add(a, b):
    return a + b"""

    return {
        "improved_draft": improved,
        "draft": improved,
        "iteration_count": state["iteration_count"] + 1
    }


def finalize_node(state: ReflectionState) -> dict:
    """Set final answer."""
    return {"final_answer": state["draft"]}


MAX_ITERATIONS = 3


def reflection_router(state: ReflectionState) -> Literal["improve", "finalize"]:
    """Route based on quality score and iteration limit."""
    if state["quality_score"] >= 8:
        return "finalize"

    if state["iteration_count"] >= MAX_ITERATIONS:
        return "finalize"

    return "improve"


# Build graph
builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_node)
builder.add_node("critique", critique_node)
builder.add_node("improve", improve_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges(
    "critique", reflection_router, {"improve": "improve", "finalize": "finalize"}
)
builder.add_edge("improve", "critique")
builder.add_edge("finalize", END)

graph = builder.compile()

# Run
result = graph.invoke(
    cast(ReflectionState, {
        "user_request": "Write Python add function",
        "draft": "",
        "critique": "",
        "improved_draft": "",
        "quality_score": 0,
        "iteration_count": 0,
        "final_answer": ""
    })
)
print(f"[Request] {result['user_request']}")
print(f"\n[Final Answer]")
print(result["final_answer"])
print(f"\n[Quality Score] {result['quality_score']}/10")
print(f"[Iterations] {result['iteration_count']}")
