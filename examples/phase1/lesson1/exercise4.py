# Commands
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.messages import HumanMessage
from rich import print

history = []
model = get_llm()

while True:
    user_input = input("You: ").strip()

    # Check for commands
    if user_input.startswith("/"):
        command = user_input[1:].lower()

        if command == "clear":
            history = []
            print("✅ Conversation cleared\n")
        elif command == "exit":
            print("👋 Goodbye!")
            break
        else:
            print(f"❌ Unknown command: {user_input}\n")
        continue

    if not user_input:
        continue

    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    print(f"AI: {response.content}\n")
    history.append(response)
