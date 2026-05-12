# Prompt template - Dynamic tutor
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from rich import print

prompt = ChatPromptTemplate.from_template("""
Explain the following topic clearly:

Teach me about {topic}.

Difficulty: {difficulty}

Teaching style: {style}
""")

model = get_llm()

messages = prompt.format_messages(  # DX way to make langchain human message
    topic="Vector embeddings", difficulty="easy", style="analogy-heavy"
)

response = model.invoke(messages)

print(response.content)
print("=" * 20)
print(messages)
