# Temperature Control
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print

prompt = ChatPromptTemplate.from_template("""
{instruction}
""")

parser = StrOutputParser()

# Create models with different temperatures
creative_model = get_llm(temperature=0.9)      # Very creative
balanced_model = get_llm(temperature=0.5)      # Balanced
strict_model = get_llm(temperature=0)          # Deterministic

# Test with same prompt across different temperatures
instruction = "Tell me an interesting fact about space in one sentence."

print("=" * 50)
print("[CREATIVE] Temperature: 0.9")
print("=" * 50)
chain = prompt | creative_model | parser
response = chain.invoke({"instruction": instruction})
print(response)

print("\n" + "=" * 50)
print("[BALANCED] Temperature: 0.5")
print("=" * 50)
chain = prompt | balanced_model | parser
response = chain.invoke({"instruction": instruction})
print(response)

print("\n" + "=" * 50)
print("[STRICT] Temperature: 0.0")
print("=" * 50)
chain = prompt | strict_model | parser
response = chain.invoke({"instruction": instruction})
print(response)
