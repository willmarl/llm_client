# what is this LLM client?

**TLDR**: abstract langchain even more. mainly 1 line functions.

`response = get_llm().invoke("Why is the sky blue")`
`caption = generate_image_caption("data/cat.jpg")`

---

This is suppose to be what i imagined langchain was. A convinent way to do all sorts of AI stuff like talk to LLM, load images, easily CRUD. (future, need to add agaents, memory, async, other stuff i dont know about)
langchain is more of centralizer for all popular AI's out there. the goal of same code to work with openai, anthropic, ollama, etc.
Still need to do lots of code for loading LLM, cast text, imgs, file locations, whatever to be langchain's `doc` type.
i wanted to be able to do simple stuff like
_"heres my task. heres a file of everything about me. use this prompt "You are a ...". go forth and do my bidding"_

1. does RAG
2. send job to LLM
3. file, api call, whatever is made

---

# install/setup

In root project, make sure you `.env` has what`.env.template` contains.
Change to your api keys, models, file locations, etc

for more info about [env setup](./env.md)

```ini
# LLM settings
LLM_PROVIDER=openai     # openai | anthropic | ollama
LLM_MODEL=gpt-4o-mini   # cheapest

# Text Embeddings settings
TEXT_EMBEDDINGS_PROVIDER=openai   # openai | ollama
TEXT_EMBEDDINGS_MODEL=text-embedding-3-small   # for OpenAI
# Example Ollama model: nomic-embed-text

# Image model settings
IMAGE_MODEL_PROVIDER=openai # openai | ollama | anthropic
IMAGE_MODEL=gpt-4.1 # gpt-4.1 | llava:7b

# Image embeddings settings
IMAGE_EMBEDDINGS_PROVIDER=replicate   # local | replicate
IMAGE_EMBEDDINGS_MODEL=ViT-B/32  # assuming CLIP
REPLICATE_MODEL=openai/clip

# API keys / hosts
OPENAI_API_KEY=sk-proj-xxx
ANTHROPIC_API_KEY=anth-xxx
OLLAMA_HOST=http://localhost:11434
REPLICATE_API_TOKEN=r8_xxx

# = OPTIONAL =
# VECTOR_DB_LOCATION = "foo/myVecDBstore"
# IMAGE_CAPTION_PROMPT = "Describe this image in detail"
```

> Note some fields are optional if you know what your working with. ex: text only (can omit mulitmodal/image) or ollama only (can omit non-ollama models/api keys)

Whilst in root project install `./llm/requirements.txt` (friendly reminder to use venv)
Torch is not in requirements.txt due to how big it is. should install if using CLIP locally.

---

# features

## llm

- [LLM](llm/llm.md)
- [LLM image (multimodal)](llm/image.md)

in config you can have same model for LLM, and multimodal as:

> all multimodal are LLM but not all LLM are multimodal

### image captioning

_all children of LLM image (multimodal)_

- gen image captioninggetter
- OCR text extract

## embeddings

### [text embedding](embedding/text-embed.md)

- text embedding

### [image emebdding](embedding/image-embed.md)

- clip image embedding

### [ultimate image extractor](embedding/ultimate-image-extractor.md)

_can toggle options of Captioning, CLIP, OCR, or ALL_

## [loaders](loaders.md)

- single file to doc
- folder to doc
- image
- images

## [splitters](splitters.md)

- recursive mode
- token mode

---

## [Database (chroma)](chromaClient.md)

- auto init DB
- CRUD methods for DB (ingest)

### ingest

WIP
