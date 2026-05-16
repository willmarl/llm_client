# CLI Chatbot with conversation history, system prompts, streaming, and commands
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from llm_client.config import LLM_PROVIDER, LLM_MODEL
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from rich import print

# ============================================
# STATE MANAGEMENT
# ============================================

history: list[BaseMessage] = [
    SystemMessage(
        content="You are a concise senior software engineer. Do not respond in any markdown text"
    )
]
model = get_llm()
current_system_prompt = "You are a concise senior software engineer. Do not respond in any markdown text"
current_model = f"{LLM_PROVIDER}:{LLM_MODEL}"


# ============================================
# HELPER FUNCTIONS
# ============================================

def update_system_prompt(new_prompt: str) -> None:
    """Update the system prompt in conversation history."""
    global history, current_system_prompt
    history[0] = SystemMessage(content=new_prompt)
    current_system_prompt = new_prompt
    print(f"✅ System prompt updated\n")


def switch_model(model_name: str) -> None:
    """Switch to a different model."""
    global model, current_model
    try:
        # Parse model_name format: "provider:model" or just "model"
        if ":" in model_name:
            provider, model_part = model_name.split(":", 1)
        else:
            provider = LLM_PROVIDER
            model_part = model_name

        # Dynamically import and create the model
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_ollama import ChatOllama
        from llm_client.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_HOST

        if provider == "openai":
            model = ChatOpenAI(model=model_part, api_key=OPENAI_API_KEY)
        elif provider == "anthropic":
            model = ChatAnthropic(
                model_name=model_part, api_key=ANTHROPIC_API_KEY, timeout=None, stop=None
            )
        elif provider == "ollama":
            model = ChatOllama(model=model_part, base_url=OLLAMA_HOST)
        else:
            print(f"❌ Unknown provider: {provider}\n")
            return

        current_model = f"{provider}:{model_part}"
        print(f"✅ Model switched to {current_model}\n")
    except Exception as e:
        print(f"❌ Error switching model: {e}\n")


def clear_history() -> None:
    """Clear conversation history but keep system prompt."""
    global history
    system_msg = history[0]
    history = [system_msg]
    print("✅ Conversation history cleared\n")


def show_status() -> None:
    """Display current chatbot status."""
    print(f"🤖 Current Model: {current_model}")
    print(f"📝 System Prompt: {current_system_prompt[:50]}...")
    print(f"💾 Messages in history: {len(history) - 1}\n")


def parse_command(user_input: str) -> tuple[bool, str]:
    """
    Parse and execute commands. Returns (is_command, response_message).
    If is_command=True, the input was a command and shouldn't be sent to the model.
    """
    if not user_input.startswith("/"):
        return False, ""

    command_parts = user_input[1:].split(maxsplit=1)
    command = command_parts[0].lower()
    args = command_parts[1] if len(command_parts) > 1 else ""

    if command == "model":
        if args:
            switch_model(args)
        else:
            print("Usage: /model <provider:model_name> or /model <model_name>\n")
    elif command == "system":
        if args:
            update_system_prompt(args)
        else:
            print("Usage: /system <new system prompt>\n")
    elif command == "clear":
        clear_history()
    elif command == "status":
        show_status()
    elif command == "exit":
        return True, "exit"
    else:
        print(f"❌ Unknown command: /{command}\n")
        print("Available commands:")
        print("  /model <model_name>  - Switch model")
        print("  /system <prompt>     - Update system prompt")
        print("  /clear               - Clear conversation history")
        print("  /status              - Show current status")
        print("  /exit                - Exit chatbot\n")

    return True, ""


# ============================================
# MAIN CHAT LOOP
# ============================================

if __name__ == "__main__":
    print("🚀 CLI Chatbot Started")
    print("Type '/exit' to quit or '/help' for commands\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check if it's a command
            is_command, cmd_result = parse_command(user_input)
            if is_command:
                if cmd_result == "exit":
                    print("👋 Goodbye!")
                    break
                continue

            # Process as normal message
            history.append(HumanMessage(content=user_input))

            # Stream response
            print("AI: ", end="", flush=True)
            response_content = ""
            for chunk in model.stream(history):
                if isinstance(chunk.content, str):
                    print(chunk.content, end="", flush=True)
                    response_content += chunk.content
            print()  # newline after streaming

            # Add response to history
            history.append(AIMessage(content=response_content))

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
