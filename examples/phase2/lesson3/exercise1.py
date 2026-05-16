# Async Runnable Chains
import sys
import asyncio
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

chain = prompt | model | parser

# Synchronous example
print("SYNCHRONOUS")
print("=" * 50)
response = chain.invoke({"topic": "Vector databases"})
print(response)

# Asynchronous example
print("\n\nASYNCHRONOUS")
print("=" * 50)

async def progress_indicator():
    """Show progress while waiting for response"""
    i = 0
    while True:
        print(f"\rWaiting for response... {i} ", end="", flush=True)
        await asyncio.sleep(0.3)
        i += 1

async def main():
    """Run chain asynchronously with progress indicator"""
    # Start the chain request
    chain_task = chain.ainvoke({"topic": "Docker containers"})

    # Run progress indicator while waiting
    progress_task = asyncio.create_task(progress_indicator())

    # Wait for response
    response = await chain_task

    # Stop progress indicator
    progress_task.cancel()
    try:
        await progress_task
    except asyncio.CancelledError:
        pass

    # Display result
    print("\r" + " " * 30 + "\r", end="")  # Clear progress line
    print(response)

asyncio.run(main())
