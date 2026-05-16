# Multiple Async Tasks with asyncio.gather()
import sys
import asyncio
import time
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
    """Run multiple chains concurrently"""
    topics = ["Machine Learning", "Quantum Computing", "Neural Networks"]

    # Define multiple tasks
    tasks = [chain.ainvoke({"topic": topic}) for topic in topics]

    # Run all tasks concurrently
    start = time.time()
    print("Running 3 tasks concurrently...")
    print("=" * 50)
    responses = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    # Display timing results
    print(f"\n✓ Completed in {elapsed:.2f}s (concurrent execution)")
    print(f"  Sequential would take ~3x longer ({elapsed * 3:.2f}s)")
    print("=" * 50)

    # Display responses
    for topic, response in zip(topics, responses):
        print(f"\n[{topic}]")
        print(response)

asyncio.run(main())
