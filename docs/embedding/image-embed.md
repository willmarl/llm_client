## Get image embedding using CLIP

> `generate_image_embeddings(img_path: str)`

```python
from rich import print
from llm_client import generate_image_embeddings

img_path = "image.jpg"
embedding_results = generate_image_embeddings(img_path)

print(type(embedding_results)) # <class 'list'>
print(f"Embedding instance: {embedding_results}")
# Embedding instance: [0.222110778093338, -0.46250641345977783, 0.49221324920654297, ...]

```

**Description**:
Give it image path and it generates list of float values using openai's CLIP

**Parameters**:

- img_path `str`: string of image path location

**Returns**:

- `List[float]` : multidimensional array
