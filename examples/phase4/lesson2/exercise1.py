# LLM use tool
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from rich import print


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""

    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""

    return a * b


model = get_llm()

# Bind tools
model_with_tools = model.bind_tools([add, multiply])

response = model_with_tools.invoke("What is 5 multiplied by 12?")
print(response.tool_calls)

# Execute requested tool
tool_map = {"add": add, "multiply": multiply}
tool_messages = []

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    tool_result = tool_map[tool_name].invoke(tool_args)
    print(tool_result)

    tool_messages.append(
        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
    )

# Continue conversation
final_response = model_with_tools.invoke(
    [("human", "What is 5 multiplied by 12?"), response, *tool_messages]
)
print(final_response.content)
