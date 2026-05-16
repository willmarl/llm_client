# Text Embedding Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import embed_single, embed_many
from rich import print

print("Text Embedding Demo")
print("=" * 50)

# Example 1: Embed single text with no prefix
print("\n[Example 1: No Prefix]")
print("-" * 50)
text1 = "Cats are independent animals"
embedding1 = embed_single(text1, prefix="none")
print(f"Text: {text1}")
print(f"Embedding dimension: {len(embedding1)}")
print(f"First 5 values: {embedding1[:5]}")

# Example 2: Embed with query prefix (from env)
print("\n[Example 2: Query Prefix (from env)]")
print("-" * 50)
query = "What makes cats good pets?"
embedding2 = embed_single(query, prefix="query")
print(f"Query: {query}")
print(f"Embedding dimension: {len(embedding2)}")
print(f"First 5 values: {embedding2[:5]}")

# Example 3: Embed with document prefix (from env)
print("\n[Example 3: Document Prefix (from env)]")
print("-" * 50)
document = "Cats are independent animals that make great companions"
embedding3 = embed_single(document, prefix="document")
print(f"Document: {document}")
print(f"Embedding dimension: {len(embedding3)}")
print(f"First 5 values: {embedding3[:5]}")

# Example 4: Custom prefix on the spot
print("\n[Example 4: Custom Prefix]")
print("-" * 50)
text4 = "Dogs are loyal and playful"
custom_prefix = "search_document: "
embedding4 = embed_single(text4, prefix="custom", custom_prefix=custom_prefix)
print(f"Text: {text4}")
print(f"Custom prefix: {custom_prefix}")
print(f"Embedding dimension: {len(embedding4)}")
print(f"First 5 values: {embedding4[:5]}")

# Example 5: Embed many texts at once
print("\n[Example 5: Embed Many Texts]")
print("-" * 50)
documents = [
    "Cats are independent",
    "Dogs are loyal",
    "Birds can fly",
    "Fish live in water"
]
embeddings_many = embed_many(documents, prefix="document")
print(f"Embedded {len(documents)} documents")
for i, (doc, emb) in enumerate(zip(documents, embeddings_many), 1):
    print(f"{i}. '{doc}' → {len(emb)} dimensions")

# Example 6: Embed multiple queries
print("\n[Example 6: Embed Multiple Queries]")
print("-" * 50)
queries = [
    "What are good pets?",
    "How do animals behave?",
    "Tell me about wildlife"
]
embeddings_queries = embed_many(queries, prefix="query")
print(f"Embedded {len(queries)} queries")
for i, (q, emb) in enumerate(zip(queries, embeddings_queries), 1):
    print(f"{i}. '{q}' → {len(emb)} dimensions")

print("\n" + "=" * 50)
print("✓ Embeddings demonstrate different prefix strategies")
print("  - 'none': No prefix")
print("  - 'query': Uses QUERY_PREFIX from env")
print("  - 'document': Uses DOCUMENT_PREFIX from env")
print("  - 'custom': Use any custom prefix on the spot")
