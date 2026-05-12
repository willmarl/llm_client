# Async Batch Processing with abatch()
import sys
import asyncio
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
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
    """Run batch processing asynchronously"""

    # Define batch inputs
    inputs = [
        {"topic": "Docker"},
        {"topic": "LangChain"},
        {"topic": "Vector DBs"}
    ]

    # Process batch asynchronously
    start = time.time()
    print("Processing batch of 3 items...")
    print("=" * 50)

    results = await chain.abatch(inputs)

    elapsed = time.time() - start
    print(f"\n✓ Completed {len(results)} items in {elapsed:.2f}s (async batch)")
    print("=" * 50)

    # Display results
    for input_data, result in zip(inputs, results):
        print(f"\n[{input_data['topic']}]")
        print(result)

asyncio.run(main())
