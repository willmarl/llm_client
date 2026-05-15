# Loop Node Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast, Literal
from langgraph.graph import StateGraph, START, END
from rich import print


class JokeState(TypedDict):
    topic: str
    joke: str
    quality_score: int
    retries: int


def generate_joke_node(state: JokeState) -> dict:
    """Generate a joke about the topic."""
    joke = f"Why did the {state['topic']} cross the road?"
    return {"joke": joke}


def evaluate_joke_node(state: JokeState) -> dict:
    """Evaluate joke quality."""
    joke = state["joke"]
    score = 9 if "road" in joke else 3
    return {"quality_score": score}


def retry_node(state: JokeState) -> dict:
    """Increment retry counter."""
    return {"retries": state["retries"] + 1}


def evaluation_router(state: JokeState) -> Literal["retry", "finalize"]:
    """Route based on quality score and retry limit."""
    if state["quality_score"] >= 7:
        return "finalize"
    if state["retries"] >= 3:
        return "finalize"
    return "retry"


# Build graph
builder = StateGraph(JokeState)
builder.add_node("generate", generate_joke_node)
builder.add_node("evaluate", evaluate_joke_node)
builder.add_node("retry", retry_node)

builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges(
    "evaluate", evaluation_router, {"retry": "retry", "finalize": END}
)
builder.add_edge("retry", "generate")

graph = builder.compile()

# Run
result = graph.invoke(
    cast(JokeState, {"topic": "chicken", "joke": "", "quality_score": 0, "retries": 0})
)
print(f"[Topic] {result['topic']}")
print(f"[Joke] {result['joke']}")
print(f"[Quality] {result['quality_score']}")
print(f"[Retries] {result['retries']}")
