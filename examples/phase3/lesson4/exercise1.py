# ChromaDB Create Demo - Adding Documents
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import Ingest, ConfigType, load_raw_text, splitter
from rich import print

print("ChromaDB Create Demo — Adding Documents to Vector Store")
print("=" * 60)

# Use temp DB location for this demo
db_location = "./lesson4_db"
config: ConfigType = {"db_location": db_location}

# Method 1: Add raw text string
print("\n[Method 1: Add Raw Text String]")
print("-" * 60)
cc = Ingest(config=config)
raw_text = "Machine learning is a subset of artificial intelligence that enables systems to learn from data"
cc.create(raw_text, rawTextType="text")
data = cc.readAll()
print(f"✓ Added raw text string")
print(f"  Documents in DB: {len(data['documents'])}")
print(f"  Sample: '{data['documents'][0][:50]}...'")

# Method 2: Add from file path
print("\n[Method 2: Add from File Path]")
print("-" * 60)
file_path = Path(__file__).parent / "sample.txt"
cc.create(str(file_path), rawTextType="path")
data = cc.readAll()
print(f"✓ Added documents from file: {file_path.name}")
print(f"  Documents in DB: {len(data['documents'])}")
for i, doc in enumerate(data['documents'], 1):
    print(f"  [{i}] '{doc[:45]}...'")

# Method 3: Add pre-chunked documents (manual control)
print("\n[Method 3: Add Pre-Chunked Documents]")
print("-" * 60)
text_to_chunk = """
Neural networks are inspired by biological neurons. They consist of interconnected layers of nodes
that process information. Deep learning uses multiple hidden layers to learn hierarchical representations.
Transformers are a modern neural network architecture that excel at sequence processing.
"""
docs = load_raw_text(text_to_chunk)
chunks = splitter(docs, chunk_size=50, chunk_overlap=10, method="recursive")
print(f"  Created {len(chunks)} chunks (chunk_size=50, overlap=10)")
cc.create(chunks)
data = cc.readAll()
print(f"✓ Added pre-chunked documents")
print(f"  Total documents in DB: {len(data['documents'])}")

# Display final DB state
print("\n" + "=" * 60)
print("Final Database State:")
print("-" * 60)
for i, doc in enumerate(data['documents'], 1):
    meta = data['metadatas'][i-1] if data['metadatas'] else {}
    print(f"{i}. '{doc[:60]}...'")
    if meta:
        print(f"   Metadata: {meta}")

print("\n" + "=" * 60)
print("ℹ️  About Adding to ChromaDB")
print("-" * 60)
print("""
Three ways to add documents:

1. **Raw Text String** (rawTextType="text")
   - Simple way to add single text snippets
   - Automatically converted to Document
   - Can be auto-split based on config

2. **File Path** (rawTextType="path")
   - Load from .txt, .pdf, .md, etc
   - Automatically parsed based on extension
   - Can be auto-split based on config

3. **Pre-Chunked Documents** (pass list)
   - Full control over chunking strategy
   - Use splitter() with custom chunk_size/overlap
   - Useful for complex text processing pipelines

By default, autoSplit=True will chunk documents.
Pass pre-chunked list to bypass auto-splitting.
""")

# Cleanup
print("\n" + "=" * 60)
print("🧹 Cleaning up temp database...")
if Path(db_location).exists():
    shutil.rmtree(db_location)
    print("✓ Cleaned up")
