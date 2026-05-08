# Imaging

## Initialize Multimodal LLM

> `get_image_llm()`

This is not really good example as this is just suppose to initialize instance. To get more of an idea how this works look at more examples below

```python
from llm_client import get_image_llm

llm = get_image_llm()

message = HumanMessage(
    content=...
    # need to b64 image and send in here
)

response = llm.invoke([message])

print(response)
```

**Description**:
Initialize Multimodal LLM for langchain, intended for Image reading. Gets the Multimodal LLM based on provider and model from config.

**Parameters**:
params

**Returns**:

- LLM instance (ChatOpenAI, ChatAnthropic, or ChatOllama)

## Make LLM generate caption of image

> `generate_image_caption(image_path: str, prompt: str | None = None) -> str:`

```python
from llm_client import generate_image_caption

caption = generate_image_caption("data/cat.jpg")
# caption = generate_image_caption(
#    "data/cat.jpg", prompt="Describe the image few sentences (1-3)", full_response=True
#)

print(caption)
```

**Description**:
Generate a text caption for an image using multimodal LLM.
Can change default prompt in config.py
exmaple:

```python
IMAGE_CAPTION_PROMPT = (
    "Describe this image in detail, including objects, colors, "
    "setting, mood, and any text visible. Be concise but thorough."
)
```

**Parameters**:

- image_path: Path to the image file
- prompt: Optional custom prompt (default from config: detailed description)
- full_response: Optional to get entire langchain object response instead of just string

**Returns**:

- str: The response from the multimodal LLM.

OR

- `<class 'langchain_core.messages.ai.AIMessage'>`

> returned `response` from 2nd example
>
> - _quick visual hierachy for **openai**, will differ with other models_

```
├─ content (str)
├─ additional_kwargs (dict)
│  └─ refusal (None)
├─ response_metadata (dict)
│  ├─ token_usage (dict)
│  │  ├─ completion_tokens (int): 55
│  │  ├─ prompt_tokens (int): 782
│  │  ├─ total_tokens (int): 837
│  │  ├─ completion_tokens_details (dict)
│  │  │  ├─ accepted_prediction_tokens (int): 0
│  │  │  ├─ audio_tokens (int): 0
│  │  │  ├─ reasoning_tokens (int): 0
│  │  │  └─ rejected_prediction_tokens (int): 0
│  │  └─ prompt_tokens_details (dict)
│  │     ├─ audio_tokens (int): 0
│  │     └─ cached_tokens (int): 0
│  ├─ model_provider (str): 'openai'
│  ├─ model_name (str): 'gpt-4.1-2025-04-14'
│  ├─ system_fingerprint (str): 'fp_d38c7f4fa7'
│  ├─ id (str): 'chatcmpl-CajGgoyJpm6zVNvvZcQPfHDqids0d'
│  ├─ service_tier (str): 'default'
│  ├─ finish_reason (str): 'stop'
│  └─ logprobs (None)
├─ id (str): 'lc_run--5275d566-cf33-4eef-86f0-d529a577993c-0'
└─ usage_metadata (dict)
   ├─ input_tokens (int): 782
   ├─ output_tokens (int): 55
   ├─ total_tokens (int): 837
   ├─ input_token_details (dict)
   │  ├─ audio (int): 0
   │  └─ cache_read (int): 0
   └─ output_token_details (dict)
      ├─ audio (int): 0
      └─ reasoning (int): 0
```

## Extract text from image via OCR

> `generate_image_ocr_text(image_path: str, languages: list[str] | None = None, strategy: str = "hi_res", infer_table_structure: bool = False) -> str`

```python
from llm_client import generate_image_ocr_text

# Basic usage (English only, no warning)
extracted_text = generate_image_ocr_text("data/receipt.jpg")

# Multi-language document (English + Spanish)
extracted_text = generate_image_ocr_text("bilingual.jpg", languages=["eng", "spa"])

# Fast processing for simple screenshots
extracted_text = generate_image_ocr_text("screenshot.png", strategy="fast")

# Extract structured tables from invoices
extracted_text = generate_image_ocr_text("invoice.jpg", infer_table_structure=True)

print(extracted_text)
```

**Description**:
Extract text from an image using OCR (Optical Character Recognition). Uses Tesseract OCR via the Unstructured library to detect and extract any readable text from images. Perfect for processing receipts, documents, screenshots, or any image containing text.

**Parameters**:

- `image_path` (str): Path to the image file to extract text from
- `languages` (list[str] | None): List of language codes for OCR. Default: `["eng"]` for English only
  - Common codes: `"eng"` (English), `"spa"` (Spanish), `"fra"` (French), `"deu"` (German), `"chi_sim"` (Chinese Simplified), `"jpn"` (Japanese), `"kor"` (Korean), `"ara"` (Arabic)
- `strategy` (str): OCR processing strategy. Default: `"hi_res"`
  - `"hi_res"`: High accuracy, slower processing
  - `"fast"`: Faster processing, lower accuracy
  - `"ocr_only"`: Only OCR, no layout detection
- `infer_table_structure` (bool): Whether to detect and preserve table structures. Default: `False`

**Returns**:

- str: All extracted text content from the image, joined with newlines

**How it works**:

1. LangChain's `UnstructuredImageLoader` wraps the image with your specified parameters
2. The `unstructured` library processes the image using the selected strategy
3. Tesseract OCR engine analyzes pixels and detects text in specified languages
4. Text is extracted and returned as a string

**Use cases**:

- Extract text from receipts, invoices, or bills
- Read text from screenshots or photos of documents
- Process scanned documents in multiple languages
- Extract structured data from tables in images
- OCR for non-English documents (Chinese, Japanese, Arabic, etc.)
