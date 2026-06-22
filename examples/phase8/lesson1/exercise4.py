# Swarm Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, List, Annotated
import operator
from llm_client import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class SwarmState(TypedDict):
    user_request: str
    candidate_solutions: Annotated[List[str], operator.add]
    best_solution: str
    completed: bool


llm = get_llm()


def coder_a(state: SwarmState) -> dict:
    prompt = f"""
    Solve this problem in Python:

    {state['user_request']}

    Focus on readability.
    """
    response = llm.invoke(prompt)
    return {"candidate_solutions": [response.content]}


def coder_b(state: SwarmState) -> dict:
    prompt = f"""
    Solve this problem in Python:

    {state['user_request']}

    Focus on performance.
    """
    response = llm.invoke(prompt)
    return {"candidate_solutions": [response.content]}


def coder_c(state: SwarmState) -> dict:
    prompt = f"""
    Solve this problem in Python:

    {state['user_request']}

    Focus on minimalism.
    """
    response = llm.invoke(prompt)
    return {"candidate_solutions": [response.content]}


def evaluator_node(state: SwarmState) -> dict:
    combined = "\n\n".join(state["candidate_solutions"])
    prompt = f"""
    Evaluate these candidate solutions.

    Select the BEST one based on:
    - correctness
    - readability
    - performance

    Solutions:

    {combined}
    """
    response = llm.invoke(prompt)
    return {
        "best_solution": response.content,
        "completed": True,
    }


# Build graph
graph = StateGraph(SwarmState)

graph.add_node("coder_a", coder_a)
graph.add_node("coder_b", coder_b)
graph.add_node("coder_c", coder_c)
graph.add_node("evaluator", evaluator_node)

graph.add_edge(START, "coder_a")
graph.add_edge(START, "coder_b")
graph.add_edge(START, "coder_c")

graph.add_edge("coder_a", "evaluator")
graph.add_edge("coder_b", "evaluator")
graph.add_edge("coder_c", "evaluator")

graph.add_edge("evaluator", END)

app = graph.compile()

# Run
result = app.invoke(
    cast(
        SwarmState,
        {
            "user_request": "Build a Python cache class",
            "candidate_solutions": [],
            "best_solution": "",
            "completed": False,
        },
    )
)

print(f"[Request] {result['user_request']}")
print(f"\n[Candidates] {len(result['candidate_solutions'])} solutions received")
for i, solution in enumerate(result["candidate_solutions"], 1):
    print(f"\n--- Candidate {i} ---")
    print(solution[:300] + ("..." if len(solution) > 300 else ""))
print(f"\n[Best Solution]\n{result['best_solution']}")
