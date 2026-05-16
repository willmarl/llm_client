# Runnable Lambda
import sys
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from rich import print


def uppercase(text: str) -> str:
    return text.upper()


upper = RunnableLambda(uppercase)

model = get_llm()

prompt = ChatPromptTemplate.from_template("""
Explain {topic} simply.
""")

parser = StrOutputParser()

chain = prompt | model | parser | upper

response = chain.invoke({"topic": "vector databases"})

print(response)
