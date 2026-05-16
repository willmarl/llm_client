# Timeout Handling with asyncio.wait_for()
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rich import print

model = get_llm()

prompt = ChatPromptTemplate.from_template("""
Answer this question: {question}
""")

parser = StrOutputParser()

chain = prompt | model | parser


async def safe_call(question, timeout=10):
    """Call the chain with timeout protection"""
    try:
        print(f"Calling chain with {timeout}s timeout...")
        result = await asyncio.wait_for(
            chain.ainvoke({"question": question}),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        return "[Timeout] Chain call exceeded the time limit"


async def main():
    """Test timeout handling"""
    print("Testing timeout handling...")
    print("=" * 50)

    # Test 1: Normal call with generous timeout
    print("\n[Test 1] Generous timeout (30s)")
    result = await safe_call("What is Python?", timeout=30)
    print(result)

    # Test 2: Strict timeout (might timeout depending on response time)
    print("\n[Test 2] Strict timeout (2s)")
    result = await safe_call("Explain artificial intelligence", timeout=2)
    print(result)

asyncio.run(main())
