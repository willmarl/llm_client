# ChromaDB Update Demo
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import Ingest, ConfigType
from rich import print

print("ChromaDB Update Demo — Modifying Documents in Vector Store")
print("=" * 60)

db_location = "./lesson4_db_update"
config: ConfigType = {"db_location": db_location}
cc = Ingest(config=config)

# Seed with intentionally imperfect data to update later
print("\n[Seeding Database with Imperfect Documents]")
print("-" * 60)
documents = [
    "Cats are ok I guess",
    "Python is basically just a scripting thing",  # vague/wrong
    "Machine learning is sorta like statistics",  # vague
    "Chroma is a database thing",                 # vague
]
for doc in documents:
    cc.create(doc, rawTextType="text")

all_data = cc.readAll()
print(f"✓ Seeded {len(all_data['documents'])} documents")
for i, doc in enumerate(all_data['documents'], 1):
    print(f"  [{i}] {doc}")

# Method 1: update_by_id — programmatic (no prompts)
print("\n[Method 1: update_by_id — Programmatic]")
print("-" * 60)
print("Use when you already have the document ID from readAll()")
print()

all_data = cc.readAll()
doc_id = all_data["ids"][0]
old_text = all_data["documents"][0]

print(f"Target ID : {doc_id[:8]}...")
print(f"Old content: '{old_text}'")
print(f"New content: 'Cats are independent and graceful animals'")
print()

cc.update_by_id(doc_id, "Cats are independent and graceful animals")

# Verify the change
updated = cc.readAll()
print(f"✓ Verified: '{updated['documents'][0]}'")

# Method 2: update — interactive (prompts user to pick + confirm)
print("\n[Method 2: update — Interactive]")
print("-" * 60)
print("Searches for matching docs, shows a numbered list, you pick which to fix.")
print("Useful when you don't know the ID and want to find by content.")
print()
print("Updating the vague Python description — follow the prompts:")
print()

cc.update(
    search_query="Python programming",
    new_content="Python is a high-level programming language known for readability and simplicity",
)

# Show final DB state
print("\n[Final Database State]")
print("-" * 60)
final = cc.readAll()
for i, doc in enumerate(final['documents'], 1):
    print(f"  [{i}] {doc}")

print("\n" + "=" * 60)
print("ℹ️  About Updating in ChromaDB")
print("-" * 60)
print("""
Two update methods:

1. **cc.update_by_id(doc_id, new_content)**
   - Low-level, no prompts
   - Get doc_id from cc.readAll()["ids"]
   - Best for scripts or when ID is already known

2. **cc.update(search_query, new_content, confirm)**
   - Interactive: searches → numbered list → you pick
   - confirm=True (default): asks "y/n" before updating
   - confirm=False: updates immediately after picking
   - Best for manual cleanup of specific documents

How update works under the hood:
  - Old embedding is replaced, not just the text
  - New content is re-embedded with the same model
  - Document ID stays the same
  - Metadata is reset to {"type": "text"}
""")

# Cleanup
print("\n" + "=" * 60)
print("🧹 Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
