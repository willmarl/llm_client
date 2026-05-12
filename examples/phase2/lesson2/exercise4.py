# RunnableLambda - Post-processing with custom functions
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from rich import print

model = get_llm()
parser = StrOutputParser()

# Create analysis chains
summary_chain = (
    ChatPromptTemplate.from_template(
        "Summarize this text in one sentence:\n\n{text}"
    )
    | model
    | parser
)

sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Rate the sentiment as positive, negative, or neutral:\n\n{text}"
    )
    | model
    | parser
)

length_chain = (
    ChatPromptTemplate.from_template(
        "How many words are in this text? Return only the number:\n\n{text}"
    )
    | model
    | parser
)

# Run all analyses in parallel
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "sentiment": sentiment_chain,
    "word_count": length_chain
})

# Define custom aggregation function
def aggregate(results):
    """Transform parallel results into a structured format"""
    return {
        "summary": results["summary"],
        "sentiment": results["sentiment"].lower(),
        "word_count": int(results["word_count"]) if results["word_count"].isdigit() else 0,
        "is_positive": "positive" in results["sentiment"].lower()
    }

# Wrap function as RunnableLambda
aggregator = RunnableLambda(aggregate)

# Pipe parallel chain into aggregator
chain = parallel_chain | aggregator

# Run the complete chain
response = chain.invoke({
    "text": "LangChain is fantastic! It makes building AI applications so much easier and faster."
})

print("Parallel Analysis Results:")
print("=" * 50)
print(response)
