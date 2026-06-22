from typing import List, Literal
from ..config import (
    OPENAI_API_KEY,
    OLLAMA_HOST,
    TEXT_EMBEDDINGS_PROVIDER,
    TEXT_EMBEDDINGS_MODEL,
    QUERY_PREFIX,
    DOCUMENT_PREFIX,
    log_print,
)
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# ============================================
# TEXT EMBEDDINGS
# ============================================


def test():
    return "hello"


def generate_text_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get text embeddings for a list of texts

    Args:
        texts: List of input strings to embed
    Returns:
        List of embedding vectors
    """
    embeddings_model = get_text_embeddings()
    log_print(f"📝 Generating embeddings for {len(texts)} texts...")
    embeddings = embeddings_model.embed_documents(texts)
    log_print(f"✅ Generated embeddings with dimension: {len(embeddings[0])}")
    return embeddings


def embed_single(
    text: str,
    prefix: Literal["none", "query", "document", "custom"] = "none",
    custom_prefix: str = "",
) -> List[float]:
    """
    Embed a single text with optional prefix.

    Args:
        text: Text to embed
        prefix: "none" (no prefix), "query" (use QUERY_PREFIX from env),
                "document" (use DOCUMENT_PREFIX from env), or "custom"
        custom_prefix: Required if prefix="custom"

    Returns:
        Embedding vector
    """
    if prefix == "custom" and not custom_prefix:
        raise ValueError("custom_prefix required when prefix='custom'")

    # Build final text
    if prefix == "none":
        final_text = text
    elif prefix == "query":
        final_text = f"{QUERY_PREFIX}{text}"
    elif prefix == "document":
        final_text = f"{DOCUMENT_PREFIX}{text}"
    else:  # custom
        final_text = f"{custom_prefix}{text}"

    embeddings_model = get_text_embeddings()
    log_print(f"📝 Embedding text (prefix={prefix}): {final_text[:50]}...")
    embedding = embeddings_model.embed_query(final_text)
    log_print(f"✅ Generated embedding with dimension: {len(embedding)}")
    return embedding


def embed_many(
    texts: List[str],
    prefix: Literal["none", "query", "document", "custom"] = "none",
    custom_prefix: str = "",
) -> List[List[float]]:
    """
    Embed multiple texts with optional prefix.

    Args:
        texts: List of texts to embed
        prefix: "none" (no prefix), "query" (use QUERY_PREFIX from env),
                "document" (use DOCUMENT_PREFIX from env), or "custom"
        custom_prefix: Required if prefix="custom"

    Returns:
        List of embedding vectors
    """
    if prefix == "custom" and not custom_prefix:
        raise ValueError("custom_prefix required when prefix='custom'")

    # Build final texts
    final_texts = []
    for text in texts:
        if prefix == "none":
            final_texts.append(text)
        elif prefix == "query":
            final_texts.append(f"{QUERY_PREFIX}{text}")
        elif prefix == "document":
            final_texts.append(f"{DOCUMENT_PREFIX}{text}")
        else:  # custom
            final_texts.append(f"{custom_prefix}{text}")

    embeddings_model = get_text_embeddings()
    log_print(f"📝 Embedding {len(texts)} texts (prefix={prefix})...")
    embeddings = embeddings_model.embed_documents(final_texts)
    log_print(f"✅ Generated embeddings with dimension: {len(embeddings[0])}")
    return embeddings


def get_text_embeddings():
    """
    Convert text to vectors

    Returns:
        Text embeddings instance (OpenAIEmbeddings or OllamaEmbeddings)
    """
    log_print(
        f"📝 Initializing text embeddings: {TEXT_EMBEDDINGS_PROVIDER} ({TEXT_EMBEDDINGS_MODEL})"
    )

    if TEXT_EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=TEXT_EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
    elif TEXT_EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbeddings(model=TEXT_EMBEDDINGS_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown embeddings provider: {TEXT_EMBEDDINGS_PROVIDER}")
