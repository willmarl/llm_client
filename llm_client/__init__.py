from .llm.llm_spawner import get_llm, run_llm

_OPTIONAL_EXPORTS = {
    "get_image_llm": ("llm_client.llm.image_spawner", "vision"),
    "generate_image_caption": ("llm_client.llm.image_spawner", "vision"),
    "generate_image_ocr_text": ("llm_client.llm.image_spawner", "ocr"),
    "generate_text_embeddings": ("llm_client.embedding.text_embedding", "base"),
    "get_text_embeddings": ("llm_client.embedding.text_embedding", "base"),
    "embed_single": ("llm_client.embedding.text_embedding", "base"),
    "embed_many": ("llm_client.embedding.text_embedding", "base"),
    "generate_image_embeddings": ("llm_client.embedding.image_embedding", "vision"),
    "ultimate_image_extractor": ("llm_client.embedding.ultimate_image_extactor", "full"),
    "load_any_file": ("llm_client.loaders", "rag"),
    "load_folder": ("llm_client.loaders", "rag"),
    "load_raw_text": ("llm_client.loaders", "base"),
    "splitter": ("llm_client.splitter", "base"),
    "db_instance": ("llm_client.chroma_client", "rag"),
    "Ingest": ("llm_client.chroma_client", "rag"),
    "ConfigType": ("llm_client.chroma_client", "rag"),
}


def __getattr__(name: str):
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module 'llm_client' has no attribute {name!r}")

    module_name, extra = _OPTIONAL_EXPORTS[name]
    try:
        from importlib import import_module

        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if extra == "base":
            raise
        raise ModuleNotFoundError(
            f"{name} requires optional dependencies. Install with: pip install 'llm_client[{extra}]'"
        ) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "get_llm",
    "run_llm",
    "get_image_llm",
    "generate_image_caption",
    "generate_image_ocr_text",
    "get_text_embeddings",
    "generate_text_embeddings",
    "embed_single",
    "embed_many",
    "generate_image_embeddings",
    "ultimate_image_extractor",
    "load_any_file",
    "load_folder",
    "load_raw_text",
    "splitter",
    "db_instance",
    "Ingest",
    "ConfigType",
]
