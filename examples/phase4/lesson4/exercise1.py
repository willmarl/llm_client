# Tool Groups
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from rich import print

# Utility tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def get_current_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

# Knowledge tools
@tool
def search_notes(query: str) -> str:
    """Search through notes (mock)."""
    notes = {
        "python": "Python is a high-level programming language",
        "agent": "An agent autonomously uses tools to solve problems",
        "react": "ReAct pattern: think (reason) → act (use tools) → repeat"
    }
    return notes.get(query.lower(), "No notes found")

# System tools
@tool
def get_system_info() -> str:
    """Get system information (mock)."""
    return "System: Linux | Memory: 16GB | Uptime: 42 days"

# Group tools by category
utility_tools = [add, multiply, get_current_time]
knowledge_tools = [search_notes]
system_tools = [get_system_info]

all_tools = utility_tools + knowledge_tools + system_tools

# Setup
model = get_llm()
tool_map = {tool.name: tool for tool in all_tools}

# Bind all tools
print("Binding tool groups:")
print("=" * 40)
model_with_tools = model.bind_tools(all_tools)
print(f"Bound {len(all_tools)} tools: {[t.name for t in all_tools]}\n")

# Use tools in ReAct loop
messages: list[BaseMessage] = [HumanMessage(content="What is 10 times 5?")]

while True:
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_result = tool_map[tool_name].invoke(tool_call["args"])
        print(f"Executing [{tool_name}]: {tool_result}")
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

print(f"\nFinal: {response.content}")
