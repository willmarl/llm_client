# Async Streaming with astream()
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
Explain {topic} simply.
""")

parser = StrOutputParser()

chain = prompt | model | parser


async def main():
    """Stream response asynchronously, printing chunks as they arrive"""

    print("Streaming response (chunk-by-chunk)...")
    print("=" * 50)

    # Use astream() to get chunks as they arrive
    print("\n[Response]:\n")
    async for chunk in chain.astream({"topic": "Machine Learning"}):
        print(chunk, end="", flush=True)

    print("\n\n" + "=" * 50)
    print("✓ Streaming complete")

asyncio.run(main())
