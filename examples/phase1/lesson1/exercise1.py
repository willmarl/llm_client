# Basic Chat
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.messages import HumanMessage
from rich import print

history = []
model = get_llm()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    print(f"\nAI: {response.content}\n")
    history.append(response)
