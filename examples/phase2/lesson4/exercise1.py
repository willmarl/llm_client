# Basic Retry with tenacity
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_fixed
from rich import print

model = get_llm()

prompt = ChatPromptTemplate.from_template("""
Answer this question: {question}
""")

parser = StrOutputParser()

chain = prompt | model | parser


# Define retry logic
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
def call_chain(question):
    """Call the chain with automatic retry on failure"""
    print(f"Attempting to call chain for: {question}")
    return chain.invoke({"question": question})


# Test the retry mechanism
try:
    print("Running chain with retry protection...")
    print("=" * 50)
    result = call_chain("What is machine learning?")
    print("\n[Success]")
    print(result)
except Exception as e:
    print(f"\n[Failed after retries]")
    print(f"Error: {e}")
