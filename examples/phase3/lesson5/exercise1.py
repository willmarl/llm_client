# Basic Retriever Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import Ingest, ConfigType
from rich import print

print("Basic Retriever Demo — as_retriever() with ChromaDB")
print("=" * 60)

db_location = "./lesson5_db"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with some documents
print("\n[Seeding Database]")
print("-" * 60)
documents = [
    "Python is a high-level programming language known for readability",
    "FastAPI is a modern web framework for building APIs with Python",
    "Chroma is a vector database for storing and querying embeddings",
    "Machine learning enables computers to learn patterns from data",
    "Cats are independent animals that sleep up to 16 hours a day",
]
for doc in documents:
    cc.create(doc, rawTextType="text")

print(f"✓ Seeded {len(documents)} documents")

# Create a retriever from the vector store
print("\n[Creating Retriever]")
print("-" * 60)
print("cc.db is the raw Chroma instance — as_retriever() wraps it as a LangChain Runnable")
print()

retriever = cc.db.as_retriever()
print(f"Type: {type(retriever)}")

# Basic retrieval
print("\n[Demo 1: Basic retrieval]")
print("-" * 60)
print('retriever.invoke("python programming")')
print()

results = retriever.invoke("python programming")
for i, doc in enumerate(results, 1):
    print(f"  [{i}] {doc.page_content}")

# Custom k (number of results)
print("\n[Demo 2: Custom k — return top 2]")
print("-" * 60)
print('retriever = cc.db.as_retriever(search_kwargs={"k": 2})')
print()

retriever_top2 = cc.db.as_retriever(search_kwargs={"k": 2})
results = retriever_top2.invoke("python programming")
print(f"Returned {len(results)} results (k=2):")
for i, doc in enumerate(results, 1):
    print(f"  [{i}] {doc.page_content}")

# Compared to similarity_search — same results, different interface
print("\n[Demo 3: Retriever vs similarity_search — same results]")
print("-" * 60)
print("Both do the same lookup. The difference is the interface:")
print("  similarity_search() → plain method call, returns list[Document]")
print("  as_retriever()      → Runnable object, composable into chains")
print()

query = "vector database embeddings"
via_retriever = cc.db.as_retriever(search_kwargs={"k": 2}).invoke(query)
via_search = cc.db.similarity_search(query, k=2)

print(f'Query: "{query}"')
print()
print("Via retriever:")
for doc in via_retriever:
    print(f"  {doc.page_content}")
print()
print("Via similarity_search:")
for doc in via_search:
    print(f"  {doc.page_content}")

print("\n" + "=" * 60)
print("ℹ️  Key Takeaway")
print("-" * 60)
print("""
retriever = cc.db.as_retriever()

  - Wraps the vector store as a LangChain Runnable
  - retriever.invoke(query) works the same as similarity_search()
  - The point: chains expect a Runnable, not a raw vector store

    chain = retriever | prompt | llm   ← this only works because
                                         retriever is a Runnable

Next lesson: plug this retriever into an actual RAG chain.
""")

# Cleanup
print("=" * 60)
print("Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
