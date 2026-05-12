# Multi message templates - Persona Switching
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from rich import print

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a tutor that teaches in {style} style. Explain at {difficulty} level difficulty.""",
        ),
        ("human", "Explain {topic}"),
    ]
)

model = get_llm()

# Format messages with all required variables
messages = prompt.format_messages(
    topic="Vector embeddings", difficulty="easy", style="analogy-heavy"
)

response = model.invoke(messages)

print(response.content)
