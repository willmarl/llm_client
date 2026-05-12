# Query Rewriting Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import Ingest, ConfigType, get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rich import print

print("Query Rewriting Demo — Better Queries = Better Retrieval")
print("=" * 60)

db_location = "./lesson6_db_rewrite"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with technical docs that use specific terminology
print("\n[Seeding Database with Technical Docs]")
print("-" * 60)
documents = [
    "Kubernetes horizontal pod autoscaling adjusts replica count based on CPU or memory metrics.",
    "Docker containers isolate application processes using Linux namespaces and cgroups.",
    "PostgreSQL connection pooling via PgBouncer reduces database connection overhead.",
    "Redis pipeline batches multiple commands into a single round-trip to reduce latency.",
    "FastAPI uses async request handlers to handle high concurrency without blocking threads.",
    "JWT tokens encode claims as a signed JSON payload for stateless authentication.",
    "NGINX acts as a reverse proxy, distributing traffic across upstream application servers.",
]
for doc in documents:
    cc.create(doc, rawTextType="text")

print(f"✓ Seeded {len(documents)} technical documents")

# Build the rewrite chain
llm = get_llm(temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer. Rewrite the user's vague question into a precise, \
technical query that will retrieve better results from a documentation vector database.

Only output the rewritten query. No explanation.

Original query: {query}
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

retriever = cc.db.as_retriever(search_kwargs={"k": 2})


def retrieve_and_show(query: str):
    results = retriever.invoke(query)
    print(f"  Query   : \"{query}\"")
    for i, doc in enumerate(results, 1):
        print(f"  Result {i}: {doc.page_content}")


# Demo queries — vague things a user might actually type
vague_queries = [
    "How does scaling work?",
    "How do APIs stay fast?",
    "How does login work without sessions?",
]

for vague in vague_queries:
    print(f"\n[Query: \"{vague}\"]")
    print("-" * 60)

    rewritten = rewrite_chain.invoke({"query": vague})

    print("Before rewrite:")
    retrieve_and_show(vague)
    print()
    print(f"Rewritten → \"{rewritten}\"")
    print()
    print("After rewrite:")
    retrieve_and_show(rewritten)

print("\n" + "=" * 60)
print("Why this matters:")
print("-" * 60)
print("""
Users write vague queries. Your docs use precise terminology.
The embedding gap between "How does scaling work?" and
"Kubernetes horizontal pod autoscaling" can be large enough
to miss the right chunk entirely.

Rewriting before retrieval bridges that gap.

In production this is often done automatically before the
user's query ever hits the vector store.
""")

# Cleanup
print("=" * 60)
print("Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
