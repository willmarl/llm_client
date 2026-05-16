# Reusable Runnable Sequence
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print

model = get_llm()

prompt = ChatPromptTemplate.from_template("""
Explain {topic} simply.
""")

parser = StrOutputParser()

# Create a reusable chain
explainer_chain = prompt | model | parser

# Invoke method
## Use it multiple times with different topics
print("-" * 20)
response1 = explainer_chain.invoke({"topic": "Vector databases"})
print(response1)
print("-" * 20)
response2 = explainer_chain.invoke({"topic": "Docker"})
print(response2)
print("-" * 20)
response3 = explainer_chain.invoke({"topic": "Kubernetes"})
print(response3)

print("=" * 20)
# Batch method
topics = [
    {"topic": "Docker"},
    {"topic": "Kubernetes"},
    {"topic": "Python"},
]
## Process all at once
responses = explainer_chain.batch(topics)

for response in responses:
    print(response)
    print("=" * 20)

print("=" * 20)
# Stream method
## Process one input and stream chunks as they arrive
print("Streaming response:")
for chunk in explainer_chain.stream({"topic": "GraphQL"}):
    print(chunk, end="", flush=True)
print("\n" + "=" * 20)
