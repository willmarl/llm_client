# ReAct Loop
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from rich import print


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]
tool_map = {tool.name: tool for tool in tools}

model = get_llm()
model_with_tools = model.bind_tools(tools)

messages: list[BaseMessage] = [HumanMessage(content="What is 5 times 12 plus 20?")]

while True:
    response = model_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        break

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_result = tool_map[tool_name].invoke(tool_call["args"])
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

print(response.content)
