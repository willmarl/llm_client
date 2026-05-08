## Initalize text embedding instance

> `get_text_embeddings()`

```python
from rich import print
from llm_client import get_text_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Sample documents to store in Chroma
documents = [
    Document(
        page_content="Cats are independent animals that make great pets.",
        metadata={"source": "cats.txt"},
    ),
    Document(
        page_content="Dogs are loyal companions who love to play fetch.",
        metadata={"source": "dogs.txt"},
    ),
    Document(
        page_content="Parrots can live for decades and learn to mimic speech.",
        metadata={"source": "birds.txt"},
    ),
]

# Get the embeddings function instance for Chroma
embeddings = get_text_embeddings()

# Create a Chroma vector store with your documents
print("🔧 Creating Chroma vector store...")
vectorstore = Chroma.from_documents(
    documents=documents, embedding=embeddings, collection_name="my_collection"
)

# Query the vector store
print("\n🔍 Querying vector store...")
query = "Tell me about pets that are independent"
results = vectorstore.similarity_search(query, k=2)

print(f"\n📊 Top {len(results)} results for query: '{query}'")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content}")
    print(f"   Source: {doc.metadata['source']}")

```

**Description**:
Spawns instance of langchain text embedder for chroma DB in. Gets the text embedder from config.

**Parameters**:

- None

**Returns**:

- Provider embedding instance (ChatOpenAI, or ChatOllama)

  _Anthropic of time of writing this doesn't have text embedder_

> - Type `<class 'langchain_openai.embeddings.base.OpenAIEmbeddings'>`
> - _quick visual hierachy for **openai**, may differ with other models_
> - Embedding instance

```
├─ client (Embeddings): <openai.resources.embeddings.Embeddings object at 0x7f00c5c52e40>
├─ async_client (AsyncEmbeddings): <openai.resources.embeddings.AsyncEmbeddings object at 0x7f00c5c53770>
├─ model (str): 'text-embedding-3-small'
├─ dimensions (None)
├─ deployment (str): 'text-embedding-ada-002'
├─ openai_api_version (None)
├─ openai_api_base (None)
├─ openai_api_type (None)
├─ openai_proxy (None)
├─ embedding_ctx_length (int): 8191
├─ openai_api_key (SecretStr): '**********'
├─ openai_organization (None)
├─ allowed_special (None)
├─ disallowed_special (None)
├─ chunk_size (int): 1000
├─ max_retries (int): 2
├─ request_timeout (None)
├─ headers (None)
├─ tiktoken_enabled (bool): True
├─ tiktoken_model_name (None)
├─ show_progress_bar (bool): False
├─ model_kwargs (dict): {}
├─ skip_empty (bool): False
├─ default_headers (None)
├─ default_query (None)
├─ retry_min_seconds (int): 4
├─ retry_max_seconds (int): 20
├─ http_client (None)
├─ http_async_client (None)
└─ check_embedding_ctx_length (bool): True
```

## Generate embedding of text

> `def generate_text_embeddings(texts: List[str]) -> List[List[float]]`

```python
from llm_client import generate_text_embeddings

texts = ["Hello, world!", "This is a test."]
embeddings = generate_text_embeddings(texts)

print(f"Generated {len(embeddings)} embeddings.")
print(embeddings[0][:2])  # Print first 2 dimensions of first embedding
```

**Description**:
Input list of string to get list of float values (may be called vector). Intended to be used after chunking/splitting but can be used for singular string item.

> Note: Assumes using chroma

**Parameters**:

- texts `list[str]`: a list of strings

**Returns**:

- `List[float]` : multidimensional array
