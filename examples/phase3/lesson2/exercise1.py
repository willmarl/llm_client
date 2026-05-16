# Document Chunking Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import load_any_file, splitter
from rich import print

# Load a document from the previous lesson
current_dir = Path(__file__).parent
txt_file = current_dir.parent / "lesson1" / "cats.txt"

print("Document Chunking Demo")
print("=" * 50)

# Load document
docs = load_any_file(str(txt_file))
doc = docs[0]

print(f"\nOriginal document:")
print(f"Length: {len(doc.page_content)} characters")
print(f"Preview: {doc.page_content[:100]}...\n")

# Example 1: Recursive splitting (default)
print("[Recursive Splitting]")
print("-" * 50)
chunks_recursive = splitter(doc, chunk_size=200, chunk_overlap=20, method="recursive")
print(f"Number of chunks: {len(chunks_recursive)}")
for i, chunk in enumerate(chunks_recursive):
    print(f"\n[Chunk {i + 1}]")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Content: {chunk.page_content[:150]}...")

# Example 2: Token-based splitting
print("\n\n[Token-based Splitting]")
print("-" * 50)
chunks_token = splitter(doc, chunk_size=80, chunk_overlap=10, method="token")
print(f"Number of chunks: {len(chunks_token)}")
for i, chunk in enumerate(chunks_token):
    print(f"\n[Chunk {i + 1}]")
    print(f"Content: {chunk.page_content[:150]}...")

# Example 3: Smaller chunks
print("\n\n[Smaller Chunks (100 chars)]")
print("-" * 50)
chunks_small = splitter(doc, chunk_size=100, chunk_overlap=15, method="recursive")
print(f"Number of chunks: {len(chunks_small)}")
for i, chunk in enumerate(chunks_small):
    print(f"\n[Chunk {i + 1}] ({len(chunk.page_content)} chars)")
    print(f"{chunk.page_content}")
