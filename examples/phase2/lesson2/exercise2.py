# Runnable Passthrough with Parallel Chains
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from rich import print

model = get_llm()
parser = StrOutputParser()

# Create summary chain
summary_chain = (
    ChatPromptTemplate.from_template(
        "Summarize this text:\n\n{text}"
    )
    | model
    | parser
)

# Create sentiment chain
sentiment_chain = (
    ChatPromptTemplate.from_template(
        "Analyze sentiment:\n\n{text}"
    )
    | model
    | parser
)

# Run chains in parallel while passing through original text
parallel_chain = RunnableParallel({
    "original_text": RunnablePassthrough(),  # Pass input through unchanged
    "summary": summary_chain,
    "sentiment": sentiment_chain
})

response = parallel_chain.invoke({
    "text": "LangChain makes AI pipeline composition easier."
})

print(response)
