# Specialized Agents Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast
from llm_client import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class TeamState(TypedDict):
    user_request: str
    plan: str
    code: str
    review: str
    final_output: str


llm = get_llm()


def planner_node(state: TeamState) -> dict:
    """Break the user request into implementation steps."""
    prompt = f"""
    You are a software planner.

    Break this request into steps:

    {state['user_request']}
    """
    response = llm.invoke(prompt)
    return {"plan": response.content}


def coder_node(state: TeamState) -> dict:
    """Write Python code based on the plan."""
    prompt = f"""
    You are a senior software engineer.

    Plan:
    {state['plan']}

    Write Python code.
    """
    response = llm.invoke(prompt)
    return {"code": response.content}


def reviewer_node(state: TeamState) -> dict:
    """Review the code and pass through the final output."""
    prompt = f"""
    You are a strict code reviewer.

    Review this code:

    {state['code']}

    Give:
    - strengths
    - weaknesses
    - improvements
    """
    response = llm.invoke(prompt)
    return {
        "review": response.content,
        "final_output": state["code"],
    }


# Build graph
builder = StateGraph(TeamState)
builder.add_node("planner", planner_node)
builder.add_node("coder", coder_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "coder")
builder.add_edge("coder", "reviewer")
builder.add_edge("reviewer", END)

graph = builder.compile()

# Run
result = graph.invoke(
    cast(
        TeamState,
        {
            "user_request": "Write a Python function that checks if a string is a palindrome",
            "plan": "",
            "code": "",
            "review": "",
            "final_output": "",
        },
    )
)

print(f"[Request] {result['user_request']}")
print(f"\n[Plan]\n{result['plan']}")
print(f"\n[Code]\n{result['final_output']}")
print(f"\n[Review]\n{result['review']}")
