import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_base_import_does_not_load_optional_dependencies():
    already_loaded = set(sys.modules)

    from llm_client import get_llm

    assert callable(get_llm)

    optional_modules = {
        "clip",
        "langchain_chroma",
        "langchain_community",
        "pdf2image",
        "PIL",
        "pytesseract",
        "pybase64",
        "replicate",
        "torch",
    }
    newly_loaded = set(sys.modules) - already_loaded
    assert optional_modules.isdisjoint(newly_loaded)
