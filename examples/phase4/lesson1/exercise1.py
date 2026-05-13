# Basic Tool
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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

# Inspect tool
print("Tool:", add.name)
print("Schema:", add.args_schema.model_json_schema())  # type: ignore

# Invoke tools
print("\nResults:")
print(f"add(5, 3) = {add.invoke({'a': 5, 'b': 3})}")
print(f"multiply(5, 3) = {multiply.invoke({'a': 5, 'b': 3})}")

# Collect tools
tools = [add, multiply]
print(f"\nAvailable: {[t.name for t in tools]}")

