from typing import List
from ..config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OLLAMA_HOST,
    REPLICATE_API_TOKEN,
    LLM_PROVIDER,
    LLM_MODEL,
    TEXT_EMBEDDINGS_PROVIDER,
    TEXT_EMBEDDINGS_MODEL,
    IMAGE_MODEL_PROVIDER,
    IMAGE_MODEL,
    REPLICATE_MODEL,
    IMAGE_EMBEDDINGS_PROVIDER,
    IMAGE_EMBEDDINGS_MODEL,
    IMAGE_CAPTION_PROMPT,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import base64
from langchain_core.messages import AIMessage
from langchain_community.document_loaders.image import UnstructuredImageLoader

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
    print(f"📝 Generating embeddings for {len(texts)} texts...")
    embeddings = embeddings_model.embed_documents(texts)
    print(f"✅ Generated embeddings with dimension: {len(embeddings[0])}")
    return embeddings


def get_text_embeddings():
    """
    Convert text to vectors

    Returns:
        Text embeddings instance (OpenAIEmbeddings or OllamaEmbeddings)
    """
    print(
        f"📝 Initializing text embeddings: {TEXT_EMBEDDINGS_PROVIDER} ({TEXT_EMBEDDINGS_MODEL})"
    )

    if TEXT_EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=TEXT_EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
    elif TEXT_EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbeddings(model=TEXT_EMBEDDINGS_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown embeddings provider: {TEXT_EMBEDDINGS_PROVIDER}")
