# MessagesPlaceholder - Insert conversation history
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from rich import print

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

model = get_llm()

# Build conversation history (previous messages)
history = [
    HumanMessage(content="What is Python?"),
    AIMessage(
        content="Python is a high-level programming language known for its simplicity and readability."
    ),
    HumanMessage(content="What can I use it for?"),
    AIMessage(content="Python is used for web development, data science, automation, AI, and more."),
]

# MessagesPlaceholder inserts the history list into the template
messages = prompt.format_messages(history=history, input="Give me an example")

response = model.invoke(messages)

print(response.content)
print("=" * 20)
print("Full message structure:")
print(messages)
