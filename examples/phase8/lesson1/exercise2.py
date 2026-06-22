# Supervisor Demo
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
    current_agent: str
    completed: bool


llm = get_llm()


def supervisor_node(state: TeamState) -> dict:
    """Decide which agent should act next."""
    request = state["user_request"].lower()

    if "research" in request:
        return {"current_agent": "researcher"}

    elif "code" in request:
        return {"current_agent": "coder"}

    elif state.get("code") and not state.get("review"):
        return {"current_agent": "reviewer"}

    else:
        return {"completed": True}


def route_supervisor(state: TeamState):
    if state.get("completed"):
        return END

    return state["current_agent"]


def researcher_node(state: TeamState) -> dict:
    """Gather background and summarize findings."""
    prompt = f"""
    You are a research analyst.

    Research this topic and summarize key findings:

    {state['user_request']}
    """
    response = llm.invoke(prompt)
    return {
        "plan": response.content,
        "final_output": response.content,
    }


def coder_node(state: TeamState) -> dict:
    """Write Python code based on the plan or request."""
    context = state["plan"] or state["user_request"]
    prompt = f"""
    You are a senior software engineer.

    Task:
    {context}

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
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "reviewer": "reviewer",
        END: END,
    },
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
builder.add_edge("reviewer", "supervisor")

graph = builder.compile()

# Run
requests = [
    "Research palindrome detection algorithms",
    "Write Python code for a palindrome checker",
]

for request in requests:
    print(f"\n[Request] {request}")
    print("-" * 50)

    result = graph.invoke(
        cast(
            TeamState,
            {
                "user_request": request,
                "plan": "",
                "code": "",
                "review": "",
                "final_output": "",
                "current_agent": "",
                "completed": False,
            },
        )
    )

    if result["review"]:
        print(f"\n[Code]\n{result['code']}")
        print(f"\n[Review]\n{result['review']}")
    elif result["plan"]:
        print(f"\n[Research]\n{result['plan']}")
    print(f"\n[Final Output]\n{result['final_output']}")
