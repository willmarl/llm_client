## Split langchain document into chunks

> `splitter(doc, 200, 10, "recursive")`

```python
from rich import print
from llm_client import load_any_file, splitter

file = "foo.txt"
doc = load_any_file(file)
split = splitter(doc)
print(type(split))
print(split)
```

results

```python
<class 'list'>
[
    Document(
        metadata={'source': 'foo.txt', 'type': 'txt'},
        page_content='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut'
    ),
    Document(
        metadata={'source': 'foo.txt', 'type': 'txt'},
        page_content='quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur'
    ),
    Document(
        metadata={'source': 'foo.txt', 'type': 'txt'},
        page_content='cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit
anim id est laborum.'
    )
]
```

**Description**:
Split a list of LangChain Document objects into smaller chunks.
Uses recursive character splitting by default, or token-based splitting.

1. **RecursiveCharacterTextSplitter** (default)

   - Splits by paragraph → sentence → character, _recursively_.
   - Preserves as much context as possible per chunk.
   - **Best for:** General text, markdown, most human writing.

2. **TokenTextSplitter**
   - Splits text based on token count (matching your model tokenizer, e.g. OpenAI, tiktoken).
   - **Best for:** Matching LLM context window size exactly.
   - **Downside:** Can cut sentences or paragraphs in the middle, so context may get chopped.

**Parameters**:

1. Langchain document (to clarify, not the path)
2. chunk_size: Target size of each chunk (characters or tokens). Defaults to 200.
3. chunk_overlap: Overlap between chunks. Defaults to 50.
4. method: `"recursive"` (default) or `"token"`.

Example of using different configs

```python
split = splitter(doc, 80, 10, "token")
```

Note: Chunk overlap must be less than chunk size or else will get

`ValueError: chunk_size must be greater than chunk_overlap.`

Ex: of bad args `split = splitter(doc, 80, 85)`

**Returns**:

- list of chopped up langchain documents
