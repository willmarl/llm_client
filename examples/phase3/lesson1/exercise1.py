# File Loading Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import load_any_file, load_folder
from rich import print

# Get file paths
current_dir = Path(__file__).parent
txt_file = current_dir / "cats.txt"
pdf_file = current_dir / "cats_guide.pdf"
test_folder = current_dir / "test_folder"

print("Loading Documents from Files")
print("=" * 50)

# Load text file
print("\n[Loading TXT file]")
txt_docs = load_any_file(str(txt_file))
print(f"Loaded {len(txt_docs)} document(s)")
for doc in txt_docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:100]}...")

# Load PDF file
print("\n[Loading PDF file]")
pdf_docs = load_any_file(str(pdf_file))
print(f"Loaded {len(pdf_docs)} document(s)")
for doc in pdf_docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:100]}...")

# Load folder
print("\n[Loading test_folder]")
if test_folder.exists():
    folder_docs = load_folder(str(test_folder), verbose=False)
    print(f"Loaded {len(folder_docs)} document(s)")
    for doc in folder_docs:
        print(f"Source: {doc.metadata['source']}")
        print(f"Content preview: {doc.page_content[:100]}...")
else:
    print(f"Folder not found: {test_folder}")
    folder_docs = []

# Display full content
print("\n" + "=" * 50)
print("\n[Full TXT Content]")
for doc in txt_docs:
    print(doc.page_content)

print("\n[Full PDF Content]")
for doc in pdf_docs:
    print(doc.page_content)

if folder_docs:
    print("\n[Full Folder Contents]")
    for doc in folder_docs:
        print(f"\n--- {doc.metadata['source']} ---")
        print(doc.page_content)
