# Partial templates - Reusable with pre-filled variables
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
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

# partial() pre-fills some variables, returns a new ChatPromptTemplate
# Useful when you have default values that stay the same across multiple calls
tutor_prompt = prompt.partial(style="analogy-heavy", difficulty="easy")

# Now you only need to provide the topic
messages = tutor_prompt.format_messages(topic="Vector embeddings")

response = model.invoke(messages)

print(response.content)
print("=" * 20)

# Can reuse the same partial prompt with different topics
messages2 = tutor_prompt.format_messages(topic="Neural networks")
response2 = model.invoke(messages2)

print(response2.content)
