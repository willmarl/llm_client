# ChromaDB Read Demo - Querying Documents
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import Ingest, ConfigType
from rich import print

print("ChromaDB Read Demo — Querying Documents from Vector Store")
print("=" * 60)

# Use temp DB location for this demo
db_location = "./lesson4_db_read"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed the database with diverse documents (with metadata categories)
print("\n[Seeding Database with Sample Documents]")
print("-" * 60)

# For this demo, we'll add documents with metadata categories
# Note: Chroma auto-attaches metadata like 'type' when creating from text
backend_docs = [
    "Python is a high-level programming language known for its readability and simplicity",
    "Database systems store and manage large amounts of structured data efficiently",
    "Web development involves creating interactive websites and applications for the internet",
]

ml_docs = [
    "Machine learning enables computers to learn patterns from data without explicit programming",
    "Natural language processing allows computers to understand and generate human language",
    "Computer vision systems can analyze and interpret visual information from images and videos",
    "Deep learning uses neural networks with multiple layers to learn hierarchical representations",
]

for doc in backend_docs:
    cc.create(doc, rawTextType="text")

for doc in ml_docs:
    cc.create(doc, rawTextType="text")

data = cc.readAll()
print(f"✓ Seeded database with {len(data['documents'])} total documents")
print(f"  - {len(backend_docs)} backend-focused documents")
print(f"  - {len(ml_docs)} machine learning-focused documents")

# Demo 1: Basic similarity search
print("\n[Demo 1: Basic Similarity Search]")
print("-" * 60)
query = "How do machines learn?"
results = cc.read(query, topK=3)
print(f"Query: '{query}'")
print(f"Found {len(results)} results (topK=3):\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
    print()

# Demo 2: Different query
print("[Demo 2: Different Query]")
print("-" * 60)
query = "Computer vision and image analysis"
results = cc.read(query, topK=3)
print(f"Query: '{query}'")
print(f"Found {len(results)} results (topK=3):\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
    print()

# Demo 3: topK parameter effect
print("[Demo 3: topK Parameter Effect]")
print("-" * 60)
query = "programming"
print(f"Query: '{query}'")

print(f"\nWith topK=1:")
results = cc.read(query, topK=1)
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:60]}...")

print(f"\nWith topK=3:")
results = cc.read(query, topK=3)
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:60]}...")

# Demo 4: Read all documents
print("\n[Demo 4: Read All Documents]")
print("-" * 60)
all_data = cc.readAll()
print(f"Total documents in database: {len(all_data['documents'])}")
print(f"Total embeddings generated: {len(all_data['embeddings'])}")
print(f"\nFirst 3 documents:")
for i, doc in enumerate(all_data['documents'][:3], 1):
    print(f"{i}. {doc[:55]}...")

# Demo 5: Metadata filtering
print("\n[Demo 5: Metadata Filtering]")
print("-" * 60)
print(f"Inspecting metadata of first 3 documents:")
for i, meta in enumerate(all_data['metadatas'][:3], 1):
    print(f"{i}. {meta}")

print(f"\nFiltering query: 'machine learning' with filter={{type: 'text'}}")
results_filtered = cc.read("machine learning", topK=5, filter={"type": "text"})
print(f"Found {len(results_filtered)} results:")
for i, doc in enumerate(results_filtered, 1):
    print(f"{i}. {doc.page_content[:60]}...")

print(f"\nNote: All documents have type='text' since we created them from raw text.")
print(f"Metadata filtering is useful when you have mixed document sources (files, URLs, etc.)")

print("\n" + "=" * 60)
print("ℹ️  About Querying ChromaDB")
print("-" * 60)
print("""
Two main query methods:

1. **cc.read(query, topK, filter, printResults)**
   - Semantic similarity search
   - Returns top-K most similar documents
   - Documents ranked by cosine similarity of embeddings
   - Optional filter parameter for metadata filtering
   - Default topK=3

2. **cc.readAll(filter)**
   - Retrieves entire database (or filtered subset)
   - Returns ids, documents, embeddings, and metadata
   - Useful for inspection or export
   - Can handle millions of documents with pagination

Metadata Filtering:
  - Filter by document properties using dict syntax
  - Example: filter={"type": "txt"} or filter={"source": "file.txt"}
  - Useful when you have mixed document sources
  - Filters apply BEFORE similarity ranking for read()
  - Filters apply to selection for readAll()

How it works:
  1. Query text is converted to embedding (same model as docs)
  2. Cosine similarity computed between query and all docs
  3. Top K results sorted by similarity score
  4. Original document text returned (not embeddings)

Query tips:
  - More specific queries = better results
  - Query language should match document language
  - topK lets you control precision vs recall tradeoff
  - Use filters to narrow search to relevant document sources
""")

# Cleanup
print("\n" + "=" * 60)
print("🧹 Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
