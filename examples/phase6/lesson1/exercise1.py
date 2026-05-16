# Tool Call Demo - Story Continuation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import TypedDict, cast
from llm_client import get_llm
from langgraph.graph import StateGraph, START, END
from rich import print


class StoryAgentState(TypedDict):
    user_input: str
    selected_tool: str
    tool_input: str
    tool_result: str


llm = get_llm()


def read_file_tool(path: str) -> str:
    """Read a file."""
    with open(path, "r") as f:
        return f.read()


def append_story_tool(path: str, current_story: str) -> str:
    """Append a new sentence to continue the story."""
    prompt = f"Continue this story with exactly 1 sentence:\n\n{current_story}"
    continuation = llm.invoke(prompt)
    new_story = current_story + " " + str(continuation.content)

    with open(path, "w") as f:
        f.write(new_story)

    return new_story


def tool_selector_node(state: StoryAgentState) -> dict:
    """Select a tool based on user input."""
    text = state["user_input"].lower()

    if "continue" in text or "append" in text or "next" in text:
        return {"selected_tool": "append_story"}
    else:
        return {"selected_tool": "read_story"}


def tool_executor_node(state: StoryAgentState) -> dict:
    """Execute the selected tool."""
    story_file = "examples/phase6/lesson1/story.txt"
    tool = state["selected_tool"]

    if tool == "append_story":
        current_story = read_file_tool(story_file)
        result = append_story_tool(story_file, current_story)
    else:
        result = read_file_tool(story_file)

    return {"tool_input": story_file, "tool_result": result}


# Build graph
builder = StateGraph(StoryAgentState)
builder.add_node("selector", tool_selector_node)
builder.add_node("executor", tool_executor_node)

builder.add_edge(START, "selector")
builder.add_edge("selector", "executor")
builder.add_edge("executor", END)

graph = builder.compile()

# Setup: create story file
story_file = "examples/phase6/lesson1/story.txt"
with open(story_file, "w") as f:
    f.write("Once upon a time, there was a curious robot.")

# Run
result = graph.invoke(
    cast(StoryAgentState, {
        "user_input": "Continue the story",
        "selected_tool": "",
        "tool_input": "",
        "tool_result": ""
    })
)
print(f"[Action] {result['selected_tool']}")
print(f"[Input] {result['tool_input']}")
print(f"[Story]\n{result['tool_result']}")
