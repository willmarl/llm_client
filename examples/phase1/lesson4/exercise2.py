# JSON parser
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rich import print

model = get_llm()

prompt = ChatPromptTemplate.from_template("""
Return valid JSON only.

Topic: {topic}

Format:
{{
    "summary": "...",
    "difficulty": "..."
}}
""")

parser = JsonOutputParser()

messages = prompt.invoke({"topic": "vector databases"})

response = model.invoke(messages)

parsed = parser.invoke(response)

print(response)
print("=" * 20)
print(parsed)
