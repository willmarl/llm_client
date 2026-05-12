# File Loading Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src import load_any_file, load_folder
from langchain_community.document_loaders import WebBaseLoader
from rich import print

loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")

documents = loader.load()

print(documents[0].page_content[:500])
