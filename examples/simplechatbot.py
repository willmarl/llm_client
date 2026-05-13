# santiy check file
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

messages = []
llm = get_llm()

system_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", "{text}")]
)

parser = StrOutputParser()
chain = system_prompt | llm | parser

while True:
    humanInput = input("Write a message: ")

    # For now, just use the latest input (simple version)
    response = chain.invoke({"text": humanInput})

    messages.append({"role": "user", "content": humanInput})
    messages.append({"role": "assistant", "content": response})

    print(f"Bot: {response}\n")
