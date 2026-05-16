# Simple Verbose ReAct
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

model = get_llm()
model_with_tools = model.bind_tools(tools)

tool_map = {tool.name: tool for tool in tools}

messages: list[BaseMessage] = [HumanMessage(content="What is 5 times 12 plus 20?")]

# First model call
response = model_with_tools.invoke(messages)
messages.append(response)

# execute tool
for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_result = tool_map[tool_name].invoke(tool_args)

    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

# continue loop
response = model_with_tools.invoke(messages)
messages.append(response)

# execute again
for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_result = tool_map[tool_name].invoke(tool_args)

    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

# final model call
response = model_with_tools.invoke(messages)
print(response.content)
