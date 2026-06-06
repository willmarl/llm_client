from typing import Literal, Optional, TypedDict, Union
from rich import print
from langchain_chroma import Chroma
from .embedding.text_embedding import get_text_embeddings
from .config import VECTOR_DB_LOCATION, QUERY_PREFIX, DOCUMENT_PREFIX
from .splitter import splitter
from .loaders import load_raw_text, load_any_file
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class PrefixedEmbeddings(Embeddings):
    def __init__(
        self, base: Embeddings, query_prefix: str = "", document_prefix: str = ""
    ):
        self.base = base
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([f"{self.document_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(f"{self.query_prefix}{text}")


# Load embeddings + vector store
def db_instance(
    db_location: str = VECTOR_DB_LOCATION,
    query_prefix: str = QUERY_PREFIX,
    document_prefix: str = DOCUMENT_PREFIX,
):
    print(f"🟦🗄️⏳ Creating DB instance at {db_location}, will take some time (~10sec)")
    embeddings = PrefixedEmbeddings(
        get_text_embeddings(), query_prefix, document_prefix
    )
    db = Chroma(persist_directory=db_location, embedding_function=embeddings)
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
    query_prefix: str
    document_prefix: str


class _ResolvedConfig(TypedDict):
    autoSplit: bool
    splitter: _ResolvedSplitterConfig
    db_location: str
    query_prefix: str
    document_prefix: str


configDefault: ConfigType = {
    "autoSplit": True,
    "splitter": {"chunk_overlap": 50, "chunk_size": 200, "method": "recursive"},
    "db_location": VECTOR_DB_LOCATION,
    "query_prefix": QUERY_PREFIX,
    "document_prefix": DOCUMENT_PREFIX,
}


class Ingest:
    def __init__(self, db=None, config: Union[ConfigType, None] = None):
        self.config: _ResolvedConfig = {**configDefault, **(config or {})}  # type: ignore[typeddict-item]

        if db == None:
            print(
                f"🟦No DB passed, will attempt to use DB based off configs: {self.config['db_location']}"
            )
            db = db_instance(
                self.config["db_location"],
                self.config["query_prefix"],
                self.config["document_prefix"],
            )
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

    def read(
        self,
        query: str,
        topK: int = 3,
        filter: Optional[dict] = None,
        printResults=False,
    ):
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
        self.db.update_document(
            doc_id, Document(page_content=new_content, metadata={"type": "text"})
        )
        print(f"🟦 updated document {doc_id[:8]}...")

    def update(self, search_query: str, new_content: str, confirm: bool = True):
        """
        Interactive update: search for candidates, pick one, optionally confirm, then update.

        Args:
            search_query: text to search for matching documents
            new_content: replacement text
            confirm: if True, prompt user to confirm before updating (default True)
        """
        query_embedding = self.db.embeddings.embed_query(search_query)  # type: ignore
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]  # type: ignore

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

    def create_from_embedding(
        self,
        embedding: list[float],
        doc_id: str,
        document: str = "",
        metadata: Optional[dict] = None,
    ):
        """
        Store a pre-computed embedding (e.g. from CLIP) directly, bypassing the text embedder.

        Args:
            embedding: pre-computed vector
            doc_id: unique ID for this entry
            document: human-readable label stored alongside (e.g. filename)
            metadata: optional metadata dict
        """
        kwargs = dict(embeddings=[embedding], documents=[document], ids=[doc_id])
        if metadata:
            kwargs["metadatas"] = [metadata]
        self.db._collection.add(**kwargs)
        print(f"🟦 added embedding {doc_id[:8]}...")

    def read_from_embedding(
        self,
        embedding: list[float],
        topK: int = 3,
        filter: Optional[dict] = None,
        printResults: bool = False,
    ):
        """
        Similarity search using a pre-computed query embedding.

        Args:
            embedding: pre-computed query vector (same space as stored embeddings)
            topK: number of results to return
            filter: metadata filter dict
            printResults: whether to print results to console
        """
        results = self.db._collection.query(
            query_embeddings=[embedding],
            n_results=topK,
            where=filter or None,
            include=["documents", "metadatas", "distances"],
        )
        if printResults:
            ids = results["ids"][0]
            docs = results["documents"][0]  # type: ignore
            metas = results["metadatas"][0]  # type: ignore
            distances = results["distances"][0]  # type: ignore
            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                print(f"  id: {doc_id[:8]}... dist: {dist:.4f} doc: {doc[:60]}")
                print(f"  metadata: {meta}")
        return results

    def update_from_embedding(
        self,
        query_embedding: list[float],
        new_embedding: list[float],
        new_document: str = "",
        new_metadata: Optional[dict] = None,
        confirm: bool = True,
    ):
        """
        Interactive update: find candidates by embedding similarity, pick one, then replace its embedding.

        Args:
            query_embedding: vector to search with
            new_embedding: replacement vector
            new_document: replacement label/document string
            new_metadata: replacement metadata (None keeps existing)
            confirm: if True, prompt user to confirm before updating
        """
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]  # type: ignore

        if not ids:
            print("🟥 No matching documents found")
            return

        print("Found matching documents:")
        for i, (doc_id, text) in enumerate(zip(ids, docs)):
            print(f"  [{i}] (id: {doc_id[:8]}...) {text[:60]}")

        choice = int(input("Which to update? (enter number, -1 to cancel): "))
        if choice == -1:
            print("Cancelled")
            return

        selected_id = ids[choice]
        selected_text = docs[choice]

        if confirm:
            print(f"  Old: '{selected_text[:60]}'")
            print(f"  New doc: '{new_document[:60]}'")
            proceed = input("Confirm update? (y/n): ")
            if proceed.lower() != "y":
                print("Cancelled")
                return

        self.db._collection.update(
            ids=[selected_id],
            embeddings=[new_embedding],
            documents=[new_document],
            metadatas=[new_metadata or {}],
        )
        print(f"🟦 updated embedding {selected_id[:8]}...")

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
        query_embedding = self.db.embeddings.embed_query(search_query)  # type: ignore
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]  # type: ignore

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

    def delete_from_embedding(self, query_embedding: list[float], confirm: bool = True):
        """
        Interactive delete: find candidates by embedding similarity, pick one, then delete.

        Args:
            query_embedding: vector to search with
            confirm: if True, prompt user to confirm before deleting
        """
        raw = self.db._collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],  # type: ignore
        )
        ids = raw["ids"][0]
        docs = raw["documents"][0]  # type: ignore

        if not ids:
            print("🟥 No matching documents found")
            return

        print("Found matching documents:")
        for i, (doc_id, text) in enumerate(zip(ids, docs)):
            print(f"  [{i}] (id: {doc_id[:8]}...) {text[:60]}")

        choice = int(input("Which to delete? (enter number, -1 to cancel): "))
        if choice == -1:
            print("Cancelled")
            return

        selected_id = ids[choice]
        selected_text = docs[choice]

        if confirm:
            print(f"  Deleting: '{selected_text[:60]}'")
            proceed = input("Confirm delete? (y/n): ")
            if proceed.lower() != "y":
                print("Cancelled")
                return

        self.delete_by_id(selected_id)
