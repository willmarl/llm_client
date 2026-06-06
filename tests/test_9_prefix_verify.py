"""
Verifies two things:
1. LangChain's embedding model sends text to the model as-is (no auto prefix)
2. PrefixedEmbeddings correctly prepends query/document prefixes before calling the model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.embeddings import Embeddings
from llm_client.chroma_client import PrefixedEmbeddings
from llm_client.embedding.text_embedding import get_text_embeddings

print("""
Running prefix behavior tests
""")


class RecordingEmbeddings(Embeddings):
    """Fake embeddings that record what text it receives instead of calling an API."""

    def __init__(self):
        self.received_documents: list[list[str]] = []
        self.received_queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.received_documents.append(texts)
        return [[0.0] * 3 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.received_queries.append(text)
        return [0.0] * 3


# Test 1: LangChain does not auto-prefix — embed_query == embed_documents([text])[0]
try:
    base = get_text_embeddings()
    vec_query = base.embed_query("cats")
    vec_doc = base.embed_documents(["cats"])[0]

    if vec_query == vec_doc:
        print("LangChain embed_query sends text as-is (no auto prefix) passed ✅")
    else:
        print(
            "LangChain embed_query sends text as-is (no auto prefix) failed ❌: embed_query != embed_documents([text])[0]"
        )
except Exception as e:
    print(f"LangChain auto-prefix check failed ❌: {e}")

# Test 2: PrefixedEmbeddings prepends query_prefix on embed_query
try:
    recorder = RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(
        recorder, query_prefix="search: ", document_prefix="passage: "
    )

    wrapped.embed_query("cats")

    received = recorder.received_queries[0]
    if received == "search: cats":
        print("PrefixedEmbeddings prepends query_prefix passed ✅")
    else:
        print(
            f"PrefixedEmbeddings prepends query_prefix failed ❌: got {repr(received)}"
        )
except Exception as e:
    print(f"PrefixedEmbeddings query_prefix failed ❌: {e}")

# Test 3: PrefixedEmbeddings prepends document_prefix on embed_documents
try:
    recorder = RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(
        recorder, query_prefix="search: ", document_prefix="passage: "
    )

    wrapped.embed_documents(["cats are fluffy", "dogs are loyal"])

    received = recorder.received_documents[0]
    if received == ["passage: cats are fluffy", "passage: dogs are loyal"]:
        print("PrefixedEmbeddings prepends document_prefix passed ✅")
    else:
        print(
            f"PrefixedEmbeddings prepends document_prefix failed ❌: got {repr(received)}"
        )
except Exception as e:
    print(f"PrefixedEmbeddings document_prefix failed ❌: {e}")

# Test 4: Empty prefix is a no-op
try:
    recorder = RecordingEmbeddings()
    wrapped = PrefixedEmbeddings(recorder, query_prefix="", document_prefix="")

    wrapped.embed_query("cats")
    wrapped.embed_documents(["dogs"])

    q = recorder.received_queries[0]
    d = recorder.received_documents[0][0]

    if q == "cats" and d == "dogs":
        print("Empty prefix is a no-op passed ✅")
    else:
        print(f"Empty prefix is a no-op failed ❌: query={repr(q)}, doc={repr(d)}")
except Exception as e:
    print(f"Empty prefix no-op failed ❌: {e}")
