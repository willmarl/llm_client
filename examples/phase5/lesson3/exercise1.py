# Router Node Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, Literal
from src import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class RouterState(TypedDict):
    user_input: str
    intent: str
    response: str


llm = get_llm()


def router_node(state: RouterState) -> dict:
    """Determine intent from user input."""
    text = state["user_input"].lower()

    if "code" in text or "python" in text:
        intent = "code"
    elif "math" in text or "calculate" in text:
        intent = "math"
    else:
        intent = "general"

    return {"intent": intent}


def math_node(state: RouterState) -> dict:
    """Handle math questions."""
    prompt = f"You are a math expert. Answer this: {state['user_input']}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def code_node(state: RouterState) -> dict:
    """Handle coding questions."""
    prompt = f"You are a Python expert. Answer this: {state['user_input']}"
    response = llm.invoke(prompt)
    return {"response": response.content}


def general_node(state: RouterState) -> dict:
    """Handle general questions."""
    response = llm.invoke(state["user_input"])
    return {"response": response.content}


# Route function
def route(state: RouterState) -> Literal["math", "code", "general"]:
    """Route based on intent."""
    return cast(Literal["math", "code", "general"], state["intent"])


# Build graph
builder = StateGraph(RouterState)
builder.add_node("router", router_node)
builder.add_node("math", math_node)
builder.add_node("code", code_node)
builder.add_node("general", general_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", route, {"math": "math", "code": "code", "general": "general"}
)
builder.add_edge("math", END)
builder.add_edge("code", END)
builder.add_edge("general", END)

graph = builder.compile()

# Run examples
questions = [
    "What is 5 plus 3?",
    "How do I write a Python function?",
    "What is the capital of France?",
]

for question in questions:
    print(f"\n[Q] {question}")
    print("-" * 40)
    result = graph.invoke(
        cast(RouterState, {"user_input": question, "intent": "", "response": ""})
    )
    print(f"[Intent] {result['intent']}")
    print(f"[A] {result['response'][:150]}...")
