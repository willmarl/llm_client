# chroma client

## Create chroma db instance

```python
from llm_client import db_instance

x = db_instance()
print(type(x))
print(x)
```

```
<class 'langchain_chroma.vectorstores.Chroma'>
<langchain_chroma.vectorstores.Chroma object at 0x7f069c903230>
```

**desc**: note you should prolly never have to do this since any ingest.py method does this as it needs to get DB instance
~~to prevent DB from loading embed everytime
need to create DB instance and pass into ingest CRUD
ingest does provide auto pull instance but rather not~~

can change location in `.env` or during instance creation
default location is `/chroma_store` if no env provided

```ini
VECTOR_DB_LOCATION = "foo/mystore"
```

Can give custom config for Ingest (config for Ingest methods, not for chroma stuff, kinda)

## create ingest class
this class has CRUD methods for vector db.
```python
from llm_client import Ingest, ConfigType

chromaClient = Ingest()

chromaClient.create(...)
chromaClient.read(...)
chromaClient.readAll(...)
```
you could manage multiple chroma DB at once by doing something like this
```python
from llm_client import db_instance, Ingest, ConfigType

db_a = db_instance("mydb/a")
cc_a = Ingest(db_a)

# or could create instance and class same time

config_b: ConfigType = {
    "db_location": "mydb/b"
}
cc_b = Ingest(config_b)

cc_a.create(...)
cc_b.readAll(...)
```
### config

This is default settings

```python
class SplitterConfig(TypedDict):
    chunk_size: int
    chunk_overlap: int
    method: Literal["recursive", "token"]
class ConfigType(TypedDict):
    autoSplit: bool
    splitter:SplitterConfig
    db_location: str

configDefault: ConfigType = {
    "autoSplit": True,
    "splitter": {
        "chunk_overlap": 200,
        "chunk_size": 50,
        "method": "recursive"
    }
    "db_location": VECTOR_DB_LOCATION,
}
```

to make new and pass in to making Ingest instance

```python
from llm_client import Ingest, ConfigType

customConfig: ConfigType = {
    "autoSplit": False,
    "splitter": {
        "chunk_overlap": 150,
        "chunk_size": 20,
        "method": "token"
    }
}

chromaClient = Ingest(config=customConfig)
```

## make the CRUD for chroma

*some call it __ingest__*

my commentary: langchain chroma methods are pretty good with stuff like add text and search by similarity BUT for stuff like add_image/search by image, my chroma client will use clip.

You can still use all the langchain's chroma methods. for example:

```python
from llm_client import Ingest

cc = Ingest() # cc as in chromaClient
x = cc.db.similarity_search("cat", k=2)
print(x)
```

### CREATE - add to DB

You can let ingest auto:
`doc -> split -> embed`
or do it yourself

> `cc.create(data)`

```python
from llm_client import Ingest, load_raw_text, splitter

cc = Ingest()

myNotes = load_raw_text("Cats are fluffly pets that sleep all day")
chunks = splitter(myNotes)
cc.create(chunks)
```

**Description**: You dont need to split the docs (infact even for my example, it cant even chunk cus too few text), but its just good practice.

**Parameters**:

- data: `langchain_doc`

**Returns**:

- None

There are several ways adding with `cc.create` can be modified based off config when making class instance

- **case 1**: manual docs
  You pass in data you langchain doc-ify urself

```python
myNotes = load_raw_text("Cats are fluffly pets that sleep all day")
cc.create(chunks)
```

if `autoSplit` is `False` then it will add the Docs u passed in.
Else will split based off config

- **case 2**: manual docs + split
  You pass in chunked docs and regardless of config, will add chunks to db

```python
myNotes = load_raw_text("Cats are fluffly pets that sleep all day")
chunks = splitter(myNotes)
cc.create(chunks)
```

- **case 3**: auto docs raw text

```python
myNotes = "Cats are fluffly pets that sleep all day."
cc.create(myNotes, "text")
```

will convert text to langchain doc -> if `autoSplit` is `True` then wil split based off [config](#config) -> add to vector DB

- **case 4**: auto docs file path

```python
myNotes = "british-shorthair.txt"
cc.create(myNotes, "path")
```

will convert text to langchain doc -> if `autoSplit` is `True` then wil split based off [config](#config) -> add to vector DB

### READ - query DB

> `cc.read(query, topK, filter, printResults)`

```python
cc = Ingest()

x = cc.read("cat")
print(x)
```

returns

```python
[
    Document(id=’d88...’, metadata={‘type’: ‘text’}, page_content=’Cats are fluffly pets that sleep all day’),
    Document(
        id=’e753...’,
        metadata={‘type’: ‘txt’, ‘source’: ‘british-shorthair.txt’},
        page_content=’undemanding, they’re content to observe household goings-on from a sunny windowsill. While not lap cats, they offer steady, easygoing companionship and tolerate children
and other pets with’
    ),
    Document(
        id=’dd351...’,
        metadata={‘source’: ‘british-shorthair.txt’, ‘type’: ‘txt’},
        page_content=’The British Shorthair is a chunky, teddy-bear-like cat with a dense, plush coat (especially the iconic blue variety) and round copper eyes. Calm and undemanding, they’re
content to observe household’
    )
]
```

**Description**:
queries chroma db based of query text. by default returns 3 results. supports optional metadata filtering.

**Parameters**:

- query: `str`
- topK: `int` default 3
- filter: `dict` optional - metadata filter (e.g. `{"source": "file.txt"}` or `{"type": "txt"}`)
- printResults: `bool` default false

Basic example:

```python
cc.read("cat", 1, True)
```

With filtering:

```python
# Search for "cat" but only in documents from a specific file
results = cc.read("cat", topK=3, filter={"source": "british-shorthair.txt"})

# Search with multiple filter conditions
results = cc.read("learning", topK=5, filter={"type": "txt"})
```

**Returns**:
list of langchain doc

you can also do normal langchain chroma methods by doing

## read all docs from DB

> `cc.readAll(filter)`

```python
x = cc.readAll()
print(x)
```

With filtering:

```python
# Get all documents from a specific file
all_from_file = cc.readAll(filter={"source": "british-shorthair.txt"})

# Get all documents of a specific type
txt_only = cc.readAll(filter={"type": "txt"})
```

**Description**:
Retrieves all documents from the database with optional metadata filtering.

**Parameters**:
- filter: `dict` optional - metadata filter (e.g. `{"source": "file.txt"}` or `{"type": "txt"}`)

**Returns**:
dict that contains

- list of Ids `str`
- list of embeddings that contains list of `float`
- list of langchain docs

```python
{
    'ids': [
        'd8848d73-576a-4661-a021-d0b87dd0651d',
        '21d21e8f-f1a2-46b5-a066-6e52e57e38f3',
        '986a9184-93d4-47ec-8a68-4cc6c3bd159c',
        'ab761439-f43d-4611-8670-3c1f4663cbf1',
        'a86ee9d9-b42e-447d-babf-5b1c87363bf2',
        'dd351e68-e008-43b7-87eb-9fe72e83be64',
        'e7537682-d053-4f14-9796-2630c8b1adda',
        '861f203d-16e5-4a64-8f18-9647ed36bfe2'
    ],
    'embeddings': array([[-0.01946291, -0.01640694, -0.03704343, ..., -0.02187979,
        -0.02039248,  0.02574913],
       [-0.02951412,  0.06452275, -0.04057876, ...,  0.01231855,
        -0.00826068,  0.01805881],
       [ 0.07120392,  0.03072025,  0.02527148, ...,  0.02551803,
        -0.00569533,  0.03900437],
       ...,
       [ 0.02308034, -0.0215346 , -0.05145435, ..., -0.04031649,
         0.00858102,  0.0127683 ],
       [ 0.01604096,  0.00876324, -0.01789134, ..., -0.04354905,
        -0.01082862,  0.03888401],
       [ 0.05983385,  0.019967  ,  0.01320615, ..., -0.0068817 ,
         0.00792906, -0.00308166]], shape=(8, 1536)),
    'documents': [
        'Cats are fluffly pets that sleep all day',
        'Wearing a tie can reduce blood flow to the brain by 7.5 per cent',
        'A chicken once lived for 18 months without a head',
        'A chicken once lived for 18 months without a head',
        'A horse normally has more than one horsepower.',
        'The British Shorthair is a chunky, teddy-bear-like cat with a dense, plush coat (especially the iconic blue variety) and round copper eyes. Calm and
undemanding, they’re content to observe household',
        'undemanding, they’re content to observe household goings-on from a sunny windowsill. While not lap cats, they offer steady, easygoing companionship and
tolerate children and other pets with',
        'and tolerate children and other pets with equanimity.'
    ],
    'uris': None,
    'included': ['documents', 'metadatas', 'embeddings'],
    'data': None,
    'metadatas': [
        {'type': 'text'},
        {'type': 'text'},
        {'type': 'text'},
        {'type': 'text'},
        {'type': 'text'},
        {'source': 'british-shorthair.txt', 'type': 'txt'},
        {'type': 'txt', 'source': 'british-shorthair.txt'},
        {'type': 'txt', 'source': 'british-shorthair.txt'}
    ]
}
```
