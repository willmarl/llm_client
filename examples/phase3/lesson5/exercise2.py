# RAG Chain Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import Ingest, ConfigType, get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rich import print

print("RAG Chain Demo — Retriever + Prompt + LLM")
print("=" * 60)

db_location = "./lesson5_db_rag"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with domain-specific knowledge the LLM wouldn't know on its own
print("\n[Seeding Database with Custom Knowledge]")
print("-" * 60)
documents = [
    "Project Orion uses a microservices architecture with 12 services deployed on Kubernetes.",
    "The Orion API gateway runs on port 8080 and requires Bearer token authentication.",
    "Orion's database is PostgreSQL 15 hosted on AWS RDS in us-east-1.",
    "The Orion team deploys every Friday at 2pm UTC using GitHub Actions.",
    "Orion uses Redis for session caching with a 30-minute TTL.",
]
for doc in documents:
    cc.create(doc, rawTextType="text")

print(f"✓ Seeded {len(documents)} documents")

# Build the RAG chain
print("\n[Building RAG Chain]")
print("-" * 60)

retriever = cc.db.as_retriever(search_kwargs={"k": 3})
llm = get_llm(temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context. If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("Chain built: retriever | prompt | llm | StrOutputParser")

# Run queries
print("\n[Demo 1: Question answered from context]")
print("-" * 60)
question = "What port does the Orion API gateway run on?"
print(f"Q: {question}")
print()
answer = chain.invoke(question)
print(f"A: {answer}")

print("\n[Demo 2: Another factual question]")
print("-" * 60)
question = "What database does Orion use and where is it hosted?"
print(f"Q: {question}")
print()
answer = chain.invoke(question)
print(f"A: {answer}")

print("\n[Demo 3: Question outside the context]")
print("-" * 60)
question = "What is the CEO's name?"
print(f"Q: {question}")
print()
answer = chain.invoke(question)
print(f"A: {answer}")

print("\n" + "=" * 60)
print("How the chain works:")
print("-" * 60)
print("""
chain = (
    {
        "context": retriever,       <- retriever.invoke(question) runs here
        "question": RunnablePassthrough(),  <- question passes through unchanged
    }
    | prompt    <- fills {context} and {question} into the template
    | llm       <- generates answer
    | StrOutputParser()  <- extracts the text string from the response
)

chain.invoke("your question") triggers the whole pipeline left to right.
""")

# Cleanup
print("=" * 60)
print("Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
