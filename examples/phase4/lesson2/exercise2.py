# Tool Choice Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.tools import tool
from rich import print

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = get_llm()
question = "What is 5 multiplied by 3?"

# 1. Default: model can choose to use tool or not
print("1. Default (model decides):")
print("-" * 40)
default_model = model.bind_tools([add, multiply])
response = default_model.invoke(question)
print(f"Q: {question}")
print(f"Tool called: {response.tool_calls}")
print()

# 2. tool_choice="any": model MUST call a tool
print("2. tool_choice='any' (must call something):")
print("-" * 40)
any_model = model.bind_tools([add, multiply], tool_choice="any")
response = any_model.invoke(question)
print(f"Q: {question}")
print(f"Tool called: {response.tool_calls}")
print()

# 3. tool_choice="multiply": model MUST use multiply
print("3. tool_choice='multiply' (forced to multiply):")
print("-" * 40)
forced_model = model.bind_tools([add, multiply], tool_choice="multiply")
response = forced_model.invoke("What is 5 plus 3?")
print(f"Q: What is 5 plus 3?")
print(f"Tool forced: {response.tool_calls}")
print()

print("=" * 40)
print("Key Difference:")
print("  Default  → model decides (may skip tool)")
print("  'any'    → model must call a tool")
print("  'name'   → model must call that tool")
