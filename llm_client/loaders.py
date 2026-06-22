import mimetypes
from pathlib import Path
from langchain_core.documents import Document


def load_raw_text(text: str):
    text_doc = Document(page_content=text, metadata={"type": "text"})
    return text_doc


def load_any_file(path: str | Path):
    """
    Auto-detect and load file into LangChain Documents.
    Supports: PDF, DOCX, PPTX, XLSX, CSV, HTML, MD, JSON, TXT.
    Falls back to plain UTF-8 text if possible.
    Raises ValueError for unsupported binary files.

    Always returns a list of Documents for consistency.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    try:
        from langchain_community.document_loaders import (
            PyPDFLoader,
            CSVLoader,
            UnstructuredHTMLLoader,
            UnstructuredMarkdownLoader,
            UnstructuredPowerPointLoader,
            UnstructuredWordDocumentLoader,
            UnstructuredExcelLoader,
            JSONLoader,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "load_any_file requires optional loader dependencies. Install with: pip install 'llm_client[rag]'"
        ) from exc

    mimetype, _ = mimetypes.guess_type(path)
    ext = path.suffix.lower()

    # --- Specialized loaders ---
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif mimetype == "text/csv" or ext == ".csv":
        return CSVLoader(str(path)).load()
    elif mimetype == "text/html" or ext in [".html", ".htm"]:
        return UnstructuredHTMLLoader(str(path)).load()
    elif ext == ".md":
        return UnstructuredMarkdownLoader(str(path)).load()
    elif ext in [".docx", ".doc"]:
        return UnstructuredWordDocumentLoader(str(path)).load()
    elif ext in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(str(path)).load()
    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(str(path)).load()
    elif ext == ".json":
        return JSONLoader(str(path), jq_schema=".").load()

    # --- Fallback: plain text ---
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc = Document(
            page_content=text, metadata={"source": str(path), "type": ext.lstrip(".")}
        )
        return [doc]
    except Exception:
        raise ValueError(f"Unsupported or binary file: {path}")


def load_folder(folder: str | Path, recursive: bool = True, verbose: bool = True):
    """
    Recursively load all supported files from a folder into Documents.
    Skips unsupported files gracefully.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"No such folder: {folder}")

    all_docs = []
    for path in folder.rglob("*") if recursive else folder.glob("*"):
        if path.is_file():
            try:
                docs = load_any_file(str(path))
                all_docs.extend(docs)
                if verbose:
                    print(f"Loaded {len(docs)} docs from {path}")
            except Exception as e:
                if verbose:
                    print(f"Skipping {path}: {e}")
    return all_docs
