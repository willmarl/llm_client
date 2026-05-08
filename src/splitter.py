from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Literal


def splitter(
    lang_doc: Document | List[Document],
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    method: Literal["recursive", "token"] = "recursive",
) -> List[Document]:
    """
    Split a list of LangChain Document objects into smaller chunks.

    Uses recursive character splitting by default, or token-based splitting.

    Args:
        lang_doc: Document object to split.
        chunk_size: Target size of each chunk (characters or tokens). Defaults to 200.
        chunk_overlap: Overlap between chunks. Defaults to 50.
        method: "recursive" (default) or "token".

    Returns:
        List[Document]: List of smaller Document chunks with preserved metadata.

    Raises:
        ValueError: If chunk_size <= chunk_overlap or invalid method.

    Example:
        Input:
            >>> from loaders import load_any_file
            >>> from splitter import splitter
            >>>
            >>> single_file = load_any_file("cat-breeds/bengal.txt")
            >>> splitTest = splitter(single_file)
            >>> print(splitTest)
        Output:
        [
            Document(
                metadata={'source': 'cat-breeds/bengal.txt', 'type': 'txt'},
                page_content='Bengals are wild-looking domestic cats...'
            ),
            Document(
                metadata={'source': 'cat-breeds/bengal.txt', 'type': 'txt'},
                page_content='High energy and intelligence demand active...'
            )
        ]
    """
    if chunk_size < chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    # Normalize input to always be a list
    docs = lang_doc if isinstance(lang_doc, list) else [lang_doc]

    if method == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif method == "token":
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError("Invalid method. Choose 'recursive' or 'token'.")

    chunks = splitter.split_documents(docs)
    return chunks
