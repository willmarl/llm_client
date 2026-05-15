# Planner/Executor Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, Literal
from langgraph.graph import StateGraph, START, END
from rich import print


class PlannerState(TypedDict):
    user_request: str
    plan: list[str]
    current_step: int
    current_task: str
    completed_results: list[str]
    final_output: str


def planner_node(state: PlannerState) -> dict:
    """Create a plan from user request."""
    request = state["user_request"]
    plan = [
        f"Research background of: {request}",
        f"Find key challenges about: {request}",
        f"Summarize opportunities for: {request}"
    ]
    return {"plan": plan, "current_step": 0}


def task_selector_node(state: PlannerState) -> dict:
    """Select the current task from the plan."""
    step = state["current_step"]
    task = state["plan"][step]
    return {"current_task": task}


def executor_node(state: PlannerState) -> dict:
    """Execute the current task."""
    task = state["current_task"]
    result = f"Completed: {task}"
    results = state.get("completed_results", [])
    return {"completed_results": results + [result]}


def progress_node(state: PlannerState) -> dict:
    """Move to the next step."""
    return {"current_step": state["current_step"] + 1}


def execution_router(state: PlannerState) -> Literal["select_task", "finalize"]:
    """Route based on whether all steps are complete."""
    if state["current_step"] >= len(state["plan"]):
        return "finalize"
    return "select_task"


def finalize_node(state: PlannerState) -> dict:
    """Combine all results into final output."""
    combined = "\n".join(state["completed_results"])
    return {"final_output": combined}


# Build graph
builder = StateGraph(PlannerState)
builder.add_node("planner", planner_node)
builder.add_node("select_task", task_selector_node)
builder.add_node("executor", executor_node)
builder.add_node("progress", progress_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "select_task")
builder.add_edge("select_task", "executor")
builder.add_edge("executor", "progress")
builder.add_conditional_edges(
    "progress", execution_router, {"select_task": "select_task", "finalize": "finalize"}
)
builder.add_edge("finalize", END)

graph = builder.compile()

# Run
result = graph.invoke(
    cast(PlannerState, {
        "user_request": "machine learning",
        "plan": [],
        "current_step": 0,
        "current_task": "",
        "completed_results": [],
        "final_output": ""
    })
)
print(f"[Request] {result['user_request']}")
print(f"\n[Results]")
print(result["final_output"])
