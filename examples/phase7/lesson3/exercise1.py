# Interrupt / Human-in-the-Loop Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig
from rich import print


class DeployState(TypedDict):
    deployment_plan: str
    approved: bool


def create_plan(state: DeployState) -> dict:
    return {"deployment_plan": "Deploy version 2.0 to production"}


def approval_node(state: DeployState) -> dict:
    """Pause and wait for human approval."""
    approval = interrupt({
        "question": "Approve deployment?",
        "plan": state["deployment_plan"]
    })
    return {"approved": approval}


def deploy_node(state: DeployState) -> dict:
    if state["approved"]:
        print("[Deploy] Deploying application...")
    else:
        print("[Deploy] Deployment cancelled.")
    return {}


# Build graph
builder = StateGraph(DeployState)
builder.add_node("create_plan", create_plan)
builder.add_node("approval", approval_node)
builder.add_node("deploy", deploy_node)

builder.add_edge(START, "create_plan")
builder.add_edge("create_plan", "approval")
builder.add_edge("approval", "deploy")
builder.add_edge("deploy", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = cast(RunnableConfig, {"configurable": {"thread_id": "deploy-1"}})

# Run 1: graph pauses at approval_node
print("[Run 1] Starting deployment pipeline...")
graph.invoke(
    cast(DeployState, {"deployment_plan": "", "approved": False}),
    config=config
)

# Inspect interrupt
state = graph.get_state(config)
interrupt_data = state.tasks[0].interrupts[0].value
print(f"\n[Interrupted] {interrupt_data['question']}")
print(f"[Plan] {interrupt_data['plan']}")

# Run 2: resume with human's answer
print(f"\n[Human] Approved: True")
graph.invoke(
    Command(resume=True),
    config=config
)
