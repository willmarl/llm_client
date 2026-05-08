## Convert raw text inline to langchain doc

> `load_raw_text(text)`

```python
from llm_client import load_raw_text

text = "A cat is sitting on a windowsill."
doc = load_raw_text(text)

print(doc) # Document(metadata={'type': 'text'}, page_content='A cat is sitting on a windowsill.')
print(doc.page_content) # A cat is sitting on a windowsill.
print(doc.metadata) # {'type': 'text'}
```

**Description**:
convert raw text to langchain doc

**Parameters**:

- test: `str`: inline/raw text

**Returns**:

- langchain doc

## Convert any text file to langchain document

> `load_any_file(file_path)`

```python
from rich import print
from llm_client import load_any_file

file = "foo.txt"
doc = load_any_file(file)
print(type(doc))
print(doc)
```

_print statements_:

```python
<class 'langchain_core.documents.base.Document'>
Document(
    metadata={'source': 'foo.txt', 'type': 'txt'},
    page_content='Lorem ipsum dolor sit amet...'
)
```

**Description**:
Auto-detect and load file into LangChain Documents.
Supports: PDF, DOCX, PPTX, XLSX, CSV, HTML, MD, JSON, TXT.
Falls back to plain UTF-8 text if possible.
Raises ValueError for unsupported binary files.

**Parameters**:

- file_path: `str`: file path for text files

**Returns**:

- langchain document

## Convert folder of files (if applicable) to langchain documents

> `load_folder(folder_path)`

```python
from rich import print
from llm_client import load_folder

folder = "cat-breeds"
doc = load_folder(folder)
print(type(doc))
print(doc)
```

_pritn statements_

```python
<class 'list'>
[
    Document(
        metadata={'source': 'cat-breeds/maine-coon.txt', 'type': 'txt'},
        page_content='Often called “gentle giants,” Maine Coons ...'
    ),
    Document(
        metadata={'source': 'cat-breeds/bengal.txt', 'type': 'txt'},
        page_content='Bengals are wild-looking domestic cats ...'
    ),
    ...
]
```

**Description**:
Recursively load all supported files from a folder into Documents.
Skips unsupported files gracefully.

**Parameters**:

- folder_path: `str`: file path for text files

**Returns**:

- list of langchain documents
