# MMR Retrieval Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import Ingest, ConfigType
from rich import print

print("MMR Retrieval Demo — Diversity vs Similarity")
print("=" * 60)

db_location = "./lesson6_db_mmr"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with intentionally repetitive docs to make MMR's effect obvious
print("\n[Seeding Database]")
print("-" * 60)
documents = [
    # Many similar Python docs — naive retrieval will return all of these
    "Python is a high-level programming language.",
    "Python is an interpreted, high-level language with a focus on developer experience.",
    "Python emphasizes code readability and simplicity.",
    "Python supports multiple programming paradigms including object-oriented and functional.",
    "Python has a large standard library and active open source community.",
    # Different topics
    "FastAPI is a modern Python web framework for building APIs.",
    "PostgreSQL is a powerful open-source relational database.",
    "Docker containers package applications with their dependencies.",
]
for doc in documents:
    cc.create(doc, rawTextType="text")

print(f"✓ Seeded {len(documents)} documents (5 very similar Python docs + 3 different topics)")

query = "Python programming language"

# Standard similarity retrieval
print("\n[Standard Similarity Retrieval — k=5]")
print("-" * 60)
print(f'Query: "{query}"')
print()

standard_retriever = cc.db.as_retriever(search_kwargs={"k": 5})
standard_results = standard_retriever.invoke(query)

print("Results:")
for i, doc in enumerate(standard_results, 1):
    print(f"  [{i}] {doc.page_content}")

print()
print("Notice: likely returns several near-identical Python descriptions.")

# MMR retrieval
print("\n[MMR Retrieval — k=5, fetch_k=20]")
print("-" * 60)
print(f'Query: "{query}"')
print()
print("search_type='mmr' fetches fetch_k candidates, then picks k diverse ones")
print()

mmr_retriever = cc.db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
    },
)
mmr_results = mmr_retriever.invoke(query)

print("Results:")
for i, doc in enumerate(mmr_results, 1):
    print(f"  [{i}] {doc.page_content}")

print()
print("Notice: fewer duplicate Python docs — MMR trades some relevance for diversity.")

# Side-by-side comparison
print("\n[Side-by-Side Comparison]")
print("-" * 60)
standard_texts = [doc.page_content for doc in standard_results]
mmr_texts = [doc.page_content for doc in mmr_results]

in_both = [t for t in standard_texts if t in mmr_texts]
only_standard = [t for t in standard_texts if t not in mmr_texts]
only_mmr = [t for t in mmr_texts if t not in standard_texts]

print(f"Shared results     : {len(in_both)}")
print(f"Standard-only      : {len(only_standard)}")
print(f"MMR-only (diverse) : {len(only_mmr)}")

if only_mmr:
    print()
    print("Docs MMR picked that standard missed (the diversity gain):")
    for doc in only_mmr:
        print(f"  → {doc}")

print("\n" + "=" * 60)
print("When to use MMR:")
print("-" * 60)
print("""
Use standard similarity when:
  - Your docs are already diverse
  - You want the single most relevant chunk

Use MMR when:
  - Docs have lots of similar/overlapping content
  - You want broad coverage of a topic
  - Redundant context is hurting your LLM answers

Key params:
  k        — how many results to return
  fetch_k  — how many candidates to pull before MMR scoring
             (higher fetch_k = more candidates to diversify from)
""")

# Cleanup
print("=" * 60)
print("Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
