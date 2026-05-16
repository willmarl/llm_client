# ChromaDB Delete Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import Ingest, ConfigType
from rich import print

print("ChromaDB Delete Demo — Removing Documents from Vector Store")
print("=" * 60)

db_location = "./lesson4_db_delete"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with some documents
print("\n[Seeding Database]")
print("-" * 60)
documents = [
    "Cats are independent and graceful animals",
    "Python is a high-level programming language",
    "Machine learning enables computers to learn from data",
    "Chroma is a vector database for storing embeddings",
    "This document is outdated and should be removed",
]
for doc in documents:
    cc.create(doc, rawTextType="text")

all_data = cc.readAll()
print(f"✓ Seeded {len(all_data['documents'])} documents")
for i, doc in enumerate(all_data['documents'], 1):
    print(f"  [{i}] {doc}")

# Method 1: delete_by_id — programmatic (no prompts)
print("\n[Method 1: delete_by_id — Programmatic]")
print("-" * 60)
print("Use when you already have the document ID from readAll()")
print()

all_data = cc.readAll()
target_text = "This document is outdated and should be removed"
target_idx = all_data["documents"].index(target_text)
doc_id = all_data["ids"][target_idx]

print(f"Target ID : {doc_id[:8]}...")
print(f"Deleting  : '{target_text}'")
print()

cc.delete_by_id(doc_id)

after = cc.readAll()
print(f"✓ Documents remaining: {len(after['documents'])}")
for i, doc in enumerate(after['documents'], 1):
    print(f"  [{i}] {doc}")

# Method 2: delete — interactive (prompts user to pick + confirm)
print("\n[Method 2: delete — Interactive]")
print("-" * 60)
print("Searches for matching docs, shows a numbered list, you pick which to remove.")
print()
print("Removing the Chroma document — follow the prompts:")
print()

cc.delete("chroma vector database")

# Show final DB state
print("\n[Final Database State]")
print("-" * 60)
final = cc.readAll()
print(f"Documents remaining: {len(final['documents'])}")
for i, doc in enumerate(final['documents'], 1):
    print(f"  [{i}] {doc}")

print("\n" + "=" * 60)
print("ℹ️  About Deleting from ChromaDB")
print("-" * 60)
print("""
Two delete methods:

1. **cc.delete_by_id(doc_id)**
   - Low-level, no prompts
   - Get doc_id from cc.readAll()["ids"]
   - Best for scripts or when ID is already known

2. **cc.delete(search_query, confirm)**
   - Interactive: searches → numbered list → you pick
   - confirm=True (default): asks "y/n" before deleting
   - confirm=False: deletes immediately after picking
   - Best for manually removing specific documents

Delete is permanent:
  - No undo — document and its embedding are gone
  - Document ID is freed and will not be reused
  - Use readAll() first to inspect before deleting
""")

# Cleanup
print("\n" + "=" * 60)
print("🧹 Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
