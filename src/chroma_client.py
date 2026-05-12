from typing import Literal, Optional, TypedDict, Union
from rich import print
from langchain_chroma import Chroma
from .embedding.text_embedding import get_text_embeddings
from .config import VECTOR_DB_LOCATION
from .splitter import splitter
from .loaders import load_raw_text, load_any_file
from langchain_core.documents import Document


# Load embeddings + vector store
def db_instance(db_location: str = VECTOR_DB_LOCATION):
    print(f"🟦🗄️⏳ Creating DB instance at {db_location}, will take some time (~10sec)")
    db = Chroma(persist_directory=db_location, embedding_function=get_text_embeddings())
    return db


"""
Reminder for me on how RAG works
User Question
    ↓
Embed (convert to vector)
    ↓
Search VectorDB (find similar vectors)
    ↓
Retrieve original text chunks
    ↓
LLM reads: [chunks] + [original question] → generates answer
"""


class SplitterConfig(TypedDict, total=False):
    chunk_size: int
    chunk_overlap: int
    method: Literal["recursive", "token"]


class _ResolvedSplitterConfig(TypedDict):
    chunk_size: int
    chunk_overlap: int
    method: Literal["recursive", "token"]


class ConfigType(TypedDict, total=False):
    autoSplit: bool
    splitter: SplitterConfig
    db_location: str


class _ResolvedConfig(TypedDict):
    autoSplit: bool
    splitter: _ResolvedSplitterConfig
    db_location: str


configDefault: ConfigType = {
    "autoSplit": True,
    "splitter": {"chunk_overlap": 50, "chunk_size": 200, "method": "recursive"},
    "db_location": VECTOR_DB_LOCATION,
}


class Ingest:
    def __init__(self, db=None, config: Union[ConfigType, None] = None):
        self.config: _ResolvedConfig = {**configDefault, **(config or {})}  # type: ignore[typeddict-item]

        if db == None:
            print(
                f"🟦No DB passed, will attempt to use DB based off configs: {self.config['db_location']}"
            )
            db = db_instance(self.config["db_location"])
        else:
            print(f"🟦using db {db}")

        self.db = db

    # raw text type [path, text]
    def create(
        self,
        data: Union[Document, list, str],
        rawTextType: Literal["path", "text", None] = None,
    ):
        """
        embed new docs and store them
        """
        # if auto false then need to manually convert to docs and split if u want
        # if auto need to do failsafe checks
        #   if only input docs, then split,
        #    if raw text then convert to docs then split
        # 1. text to docs
        # 2. docs -> split
        # 3. add to DB (it auto vectorize)
        if isinstance(data, str):
            if rawTextType == "text":
                data = load_raw_text(data)
            elif rawTextType == "path":
                data = load_any_file(data)
            else:
                raise ValueError(
                    "Detecting raw text or path but did not select 'path' or 'text'"
                )

        if self.config["autoSplit"] and not isinstance(data, list):
            chunks = splitter(
                data,
                self.config["splitter"]["chunk_size"],
                self.config["splitter"]["chunk_overlap"],
                self.config["splitter"]["method"],
            )
            data = chunks

        print(data)

        if not isinstance(data, list):
            data = [data]

        print("🟦 adding to DB")
        self.db.add_documents(data)

    def read(self, query: str, topK: int = 3, filter: Optional[dict] = None, printResults=False):
        """
        query relevant docs with optional metadata filtering

        Args:
            query: search text
            topK: number of results to return (default 3)
            filter: metadata filter dict, e.g. {"source": "file.txt"} or {"topic": "backend"}
            printResults: whether to print results to console
        """
        print("🟦reading from db")
        results = self.db.similarity_search(query, k=topK, filter=filter)
        if printResults:
            for r in results:
                print("Content:", r.page_content)
                print("Metadata:", end="")
                print(r.metadata)
        return results

    def readAll(self, filter: Optional[dict] = None):
        """
        Retrieve all documents with optional metadata filtering

        Args:
            filter: metadata filter dict, e.g. {"source": "file.txt"} or {"type": "txt"}
        """
        all_documents_data = self.db.get(
            # You can specify what to include, e.g., documents, metadatas, embeddings
            include=["documents", "metadatas", "embeddings"],
            limit=9999999,  # Use a very large number to get all, or implement pagination
            where=filter,  # Chroma uses 'where' parameter for filtering
        )
        return all_documents_data

    def update_by_id(self, doc_id: str, new_content: str):
        """
        Update a document by its Chroma ID. Get IDs from readAll()["ids"].
        """
        self.db.update_document(doc_id, Document(page_content=new_content, metadata={"type": "text"}))
        print(f"🟦 updated document {doc_id[:8]}...")

    def update(self, search_query: str, new_content: str, confirm: bool = True):
        """
        Interactive update: search for candidates, pick one, optionally confirm, then update.

        Args:
            search_query: text to search for matching documents
            new_content: replacement text
            confirm: if True, prompt user to confirm before updating (default True)
        """
        query_embedding = self.db.embeddings.embed_query(search_query)
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]

        if not ids:
            print("🟥 No matching documents found")
            return

        print("Found matching documents:")
        for i, (doc_id, text) in enumerate(zip(ids, docs)):
            print(f"  [{i}] (id: {doc_id[:8]}...) {text[:60]}...")

        choice = int(input("Which to update? (enter number, -1 to cancel): "))
        if choice == -1:
            print("Cancelled")
            return

        selected_id = ids[choice]
        selected_text = docs[choice]

        if confirm:
            print(f"  Old: '{selected_text[:60]}...'")
            print(f"  New: '{new_content[:60]}...'")
            proceed = input("Confirm update? (y/n): ")
            if proceed.lower() != "y":
                print("Cancelled")
                return

        self.update_by_id(selected_id, new_content)

    def delete_by_id(self, doc_id: str):
        """
        Delete a document by its Chroma ID. Get IDs from readAll()["ids"].
        """
        self.db.delete(ids=[doc_id])
        print(f"🟦 deleted document {doc_id[:8]}...")

    def delete(self, search_query: str, confirm: bool = True):
        """
        Interactive delete: search for candidates, pick one, optionally confirm, then delete.

        Args:
            search_query: text to search for matching documents
            confirm: if True, prompt user to confirm before deleting (default True)
        """
        query_embedding = self.db.embeddings.embed_query(search_query)
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]

        if not ids:
            print("🟥 No matching documents found")
            return

        print("Found matching documents:")
        for i, (doc_id, text) in enumerate(zip(ids, docs)):
            print(f"  [{i}] (id: {doc_id[:8]}...) {text[:60]}...")

        choice = int(input("Which to delete? (enter number, -1 to cancel): "))
        if choice == -1:
            print("Cancelled")
            return

        selected_id = ids[choice]
        selected_text = docs[choice]

        if confirm:
            print(f"  Deleting: '{selected_text[:60]}...'")
            proceed = input("Confirm delete? (y/n): ")
            if proceed.lower() != "y":
                print("Cancelled")
                return

        self.delete_by_id(selected_id)
