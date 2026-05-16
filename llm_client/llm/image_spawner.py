from ..config import (
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OLLAMA_HOST,
    REPLICATE_API_TOKEN,
    LLM_PROVIDER,
    LLM_MODEL,
    TEXT_EMBEDDINGS_PROVIDER,
    TEXT_EMBEDDINGS_MODEL,
    IMAGE_MODEL_PROVIDER,
    IMAGE_MODEL,
    REPLICATE_MODEL,
    IMAGE_EMBEDDINGS_PROVIDER,
    IMAGE_EMBEDDINGS_MODEL,
    IMAGE_CAPTION_PROMPT,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import base64
from langchain_core.messages import AIMessage
from langchain_community.document_loaders.image import UnstructuredImageLoader

# ============================================
# LLM FUNCTIONS
# ============================================


def get_image_llm():
    """
    Use multimodal LLM for image captioning

    Returns:
        Vision-capable LLM instance
    """
    print(f"👁️ Initializing vision LLM: {IMAGE_MODEL_PROVIDER} ({IMAGE_MODEL})")

    if IMAGE_MODEL_PROVIDER == "openai":
        return ChatOpenAI(model=IMAGE_MODEL, api_key=OPENAI_API_KEY)
    elif IMAGE_MODEL_PROVIDER == "anthropic":
        return ChatAnthropic(
            model_name=IMAGE_MODEL, api_key=ANTHROPIC_API_KEY, timeout=None, stop=None
        )
    elif IMAGE_MODEL_PROVIDER == "ollama":
        return ChatOllama(model=IMAGE_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown image model provider: {IMAGE_MODEL_PROVIDER}")


# ============================================
# IMAGE CAPTIONING (Multimodal)
# ============================================


def generate_image_caption(
    image_path: str, prompt: str = IMAGE_CAPTION_PROMPT, full_response: bool = False
) -> str | AIMessage:
    """
    Generate a text caption for an image using multimodal LLM

    Args:
        image_path: Path to the image file
        prompt: Optional custom prompt (default from config: detailed description)

    Returns:
        Generated caption text

    Example:
        >>> caption = generate_image_caption("photo.jpg")
        >>> print(caption)
        'A scenic mountain landscape with snow-capped peaks...'

        >>> caption = generate_image_caption(
        ...     "product.jpg",
        ...     prompt="List all visible objects"
        ... )
    """
    print(f"🤖 Generating caption for: {Path(image_path).name}")

    llm = get_image_llm()

    # Read image as base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Default prompt if none provided
    if prompt is None:
        prompt = (
            "Describe this image in detail, including objects, colors, "
            "setting, mood, and any text visible. Be concise but thorough."
        )

    # Create message with image
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
            {"type": "text", "text": prompt},
        ]
    )

    response = llm.invoke([message])
    print(f"✅ Caption generated successfully")
    if full_response:
        return response
    else:
        return str(response.content)


# ============================================
# OCR TEXT EXTRACTION
# ============================================


def generate_image_ocr_text(
    image_path: str,
    languages: list[str] | None = None,
    strategy: str = "hi_res",
    infer_table_structure: bool = False,
) -> str:
    """
    Extract text from an image using OCR

    Args:
        image_path: Path to the image file
        languages: List of language codes for OCR (e.g., ["eng", "spa", "fra"])
                   Default: ["eng"] for English only
                   Common codes: eng (English), spa (Spanish), fra (French),
                                deu (German), chi_sim (Chinese Simplified), jpn (Japanese)
        strategy: OCR processing strategy:
                  - "hi_res" (default): High accuracy, slower
                  - "fast": Faster processing, lower accuracy
                  - "ocr_only": Only OCR, no layout detection
        infer_table_structure: Whether to detect and preserve table structures

    Returns:
        Extracted text content

    Example:
        >>> # Basic usage (English only)
        >>> text = generate_image_ocr_text("receipt.jpg")

        >>> # Multiple languages
        >>> text = generate_image_ocr_text("receipt.jpg", languages=["eng", "spa"])

        >>> # Fast processing for simple images
        >>> text = generate_image_ocr_text("screenshot.png", strategy="fast")

        >>> # Extract table data
        >>> text = generate_image_ocr_text("invoice.jpg", infer_table_structure=True)
    """
    print(f"📄 Extracting text from: {Path(image_path).name}")

    # Set default language if not specified
    if languages is None:
        languages = ["eng"]

    loader = UnstructuredImageLoader(
        image_path,
        languages=languages,
        strategy=strategy,
        infer_table_structure=infer_table_structure,
    )
    docs = loader.load()

    extracted_text = "\n".join([doc.page_content for doc in docs])
    print(f"✅ Extracted {len(extracted_text)} characters")

    return extracted_text
