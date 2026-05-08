from typing import List
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_HOST, REPLICATE_API_TOKEN,
    LLM_PROVIDER, LLM_MODEL,
    TEXT_EMBEDDINGS_PROVIDER, TEXT_EMBEDDINGS_MODEL,
    IMAGE_MODEL_PROVIDER, IMAGE_MODEL, REPLICATE_MODEL,
    IMAGE_EMBEDDINGS_PROVIDER, IMAGE_EMBEDDINGS_MODEL,
)
import torch
import clip
from PIL import Image
import replicate
import pybase64
from pathlib import Path
import base64
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders.image import UnstructuredImageLoader

# ============================================
# LLM FUNCTIONS
# ============================================

def get_llm():
    """
    Get the LLM based on provider and model from config
    
    Returns:
        LLM instance (ChatOpenAI, ChatAnthropic, or ChatOllama)
    """
    print(f"🤖 Initializing LLM: {LLM_PROVIDER} ({LLM_MODEL})")
    
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == "anthropic":
        return ChatAnthropic(model_name=LLM_MODEL, api_key=ANTHROPIC_API_KEY, timeout=None, stop=None)
    elif LLM_PROVIDER == "ollama":
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown provider: {LLM_PROVIDER}")

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
        return ChatAnthropic(model_name=IMAGE_MODEL, api_key=ANTHROPIC_API_KEY, timeout=None, stop=None)
    elif IMAGE_MODEL_PROVIDER == "ollama":
        return ChatOllama(model=IMAGE_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown image model provider: {IMAGE_MODEL_PROVIDER}")

# ============================================
# TEXT EMBEDDINGS
# ============================================

def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get text embeddings for a list of texts
    
    Args:
        texts: List of input strings to embed
    Returns:
        List of embedding vectors
    """
    embeddings_model = get_text_embeddings_chroma()
    print(f"📝 Generating embeddings for {len(texts)} texts...")
    embeddings = embeddings_model.embed_documents(texts)
    print(f"✅ Generated embeddings with dimension: {len(embeddings[0])}")
    return embeddings

def get_text_embeddings_chroma():
    """
    Convert text to vectors
    
    Returns:
        Text embeddings instance (OpenAIEmbeddings or OllamaEmbeddings)
    """
    print(f"📝 Initializing text embeddings: {TEXT_EMBEDDINGS_PROVIDER} ({TEXT_EMBEDDINGS_MODEL})")
    
    if TEXT_EMBEDDINGS_PROVIDER == "openai":
        return OpenAIEmbeddings(model=TEXT_EMBEDDINGS_MODEL, api_key=OPENAI_API_KEY)
    elif TEXT_EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbeddings(model=TEXT_EMBEDDINGS_MODEL, base_url=OLLAMA_HOST)
    else:
        raise ValueError(f"❌ Unknown embeddings provider: {TEXT_EMBEDDINGS_PROVIDER}")

# ============================================
# IMAGE EMBEDDINGS (CLIP)
# ============================================

def _embed_image_local(path: str) -> List[float]:
    """
    Embed image using local CLIP model
    
    Args:
        path: Path to image file
        
    Returns:
        Embedding vector as list of floats
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🔄 Processing with local CLIP on {device.upper()}...")
    
    model, preprocess = clip.load(IMAGE_EMBEDDINGS_MODEL, device=device)
    image = preprocess(Image.open(path)).unsqueeze(0).to(device) # type: ignore
    
    with torch.no_grad():
        features = model.encode_image(image)
    
    features /= features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()

def _embed_image_replicate(path: str) -> List[float]:
    """
    Embed image using Replicate CLIP API
    
    Args:
        path: Path to image file
        
    Returns:
        Embedding vector as list of floats
    """
    print(f"  ☁️  Processing with Replicate CLIP...")
    
    base64_image = pybase64.b64encode(Path(path).read_bytes()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    output = replicate.run(
        REPLICATE_MODEL,
        input={"image": data_uri}
    )
    
    result: dict = output  # type: ignore
    return result["embedding"]

def get_image_embeddings(path: str) -> List[float]:
    """
    Turn image into vector
    
    Args:
        path: Path to image file
        
    Returns:
        Embedding vector as list of floats
    """
    if IMAGE_EMBEDDINGS_PROVIDER == "local":
        return _embed_image_local(path)
    elif IMAGE_EMBEDDINGS_PROVIDER == "replicate":
        return _embed_image_replicate(path)
    else:
        raise ValueError(f"❌ Unknown image embeddings provider: {IMAGE_EMBEDDINGS_PROVIDER}")

class ImageEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper for image embeddings
    """
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple image paths
        
        Args:
            texts: List of image file paths
            
        Returns:
            List of embedding vectors
        """
        print(f"🖼️  Embedding {len(texts)} images...")
        return [get_image_embeddings(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single image path
        
        Args:
            text: Image file path
            
        Returns:
            Embedding vector
        """
        return get_image_embeddings(text)

# Create a singleton instance you can import directly
get_image_embeddings_langchain = ImageEmbeddings()

# ============================================
# IMAGE LOADER
# ============================================

class ImageLoader:
    """
    Loads images as Documents (like TextLoader but for images)
    """
    
    def __init__(self, path: str):
        """
        Initialize loader
        
        Args:
            path: Path to image file or directory
        """
        self.path = Path(path)
    
    def load(self) -> List[Document]:
        """
        Load image(s) as Document objects
        
        Returns:
            List of Documents with image paths as page_content
        """
        if self.path.is_file():
            print(f"📁 Loading single image: {self.path.name}")
            return [self._create_document(self.path)]
        elif self.path.is_dir():
            print(f"📁 Loading images from directory: {self.path}")
            image_files = list(self.path.glob("*.jpg")) + \
                         list(self.path.glob("*.png")) + \
                         list(self.path.glob("*.jpeg"))
            print(f"✅ Found {len(image_files)} images")
            return [self._create_document(img) for img in image_files]
        else:
            raise ValueError(f"❌ Path not found: {self.path}")
    
    def _create_document(self, img_path: Path) -> Document:
        """
        Convert image path to Document
        
        Args:
            img_path: Path object for image
            
        Returns:
            Document with image metadata
        """
        return Document(
            page_content=str(img_path),
            metadata={
                "type": "image",
                "filename": img_path.name,
                "extension": img_path.suffix,
                "size": img_path.stat().st_size
            }
        )

# ============================================
# IMAGE CAPTIONING (Multimodal)
# ============================================

def generate_image_caption(image_path: str, prompt: str | None = None) -> str:
    """
    Generate a text caption for an image using multimodal LLM
    
    Args:
        image_path: Path to the image file
        prompt: Optional custom prompt (default: detailed description)
    
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
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            },
            {"type": "text", "text": prompt}
        ]
    )
    
    response = llm.invoke([message])
    print(f"✅ Caption generated successfully")
    return str(response.content)

# ============================================
# OCR TEXT EXTRACTION
# ============================================

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Extracted text content
        
    Example:
        >>> text = extract_text_from_image("receipt.jpg")
        >>> print(text)
        'RECEIPT #12345\\nTotal: $47.32'
    """
    print(f"📄 Extracting text from: {Path(image_path).name}")
    
    loader = UnstructuredImageLoader(image_path)
    docs = loader.load()
    
    extracted_text = "\n".join([doc.page_content for doc in docs])
    print(f"✅ Extracted {len(extracted_text)} characters")
    
    return extracted_text

def extract_text_from_images(image_paths: List[str]) -> List[Document]:
    """
    Extract text from multiple images and return as Documents
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        List of Documents with OCR'd text as page_content
        
    Example:
        >>> paths = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
        >>> docs = extract_text_from_images(paths)
        >>> print(f"Processed {len(docs)} documents")
    """
    print(f"📄 Starting OCR extraction for {len(image_paths)} images...")
    
    ocr_docs = []
    
    for i, img_path in enumerate(image_paths, 1):
        try:
            print(f"  [{i}/{len(image_paths)}] Processing {Path(img_path).name}...")
            
            loader = UnstructuredImageLoader(img_path)
            extracted = loader.load()
            
            for doc in extracted:
                doc.metadata["image_path"] = img_path
                doc.metadata["type"] = "ocr_text"
                doc.metadata["filename"] = Path(img_path).name
                ocr_docs.append(doc)
            
            print(f"    ✅ Extracted {len(extracted[0].page_content)} characters")
            
        except Exception as e:
            print(f"    ⚠️  OCR failed: {e}")
            continue
    
    print(f"✅ OCR complete! Processed {len(ocr_docs)} documents")
    return ocr_docs

# ============================================
# HYBRID PROCESSING PIPELINES
# ============================================

def process_images_with_clip_and_ocr(image_paths: List[str]) -> tuple[List[Document], List[Document]]:
    """
    Process images for both visual similarity (CLIP) and text search (OCR)
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        Tuple of (clip_documents, ocr_documents)
        
    Example:
        >>> paths = load_images_from_directory("./receipts")
        >>> clip_docs, ocr_docs = process_images_with_clip_and_ocr(paths)
        >>> print(f"Visual: {len(clip_docs)}, Text: {len(ocr_docs)}")
    """
    print("\n" + "="*60)
    print("🎨 HYBRID PROCESSING: CLIP + OCR")
    print("="*60 + "\n")
    
    # CLIP documents (paths only)
    print("🖼️  Step 1/2: Creating CLIP documents...")
    clip_docs = []
    for img_path in image_paths:
        doc = Document(
            page_content=img_path,
            metadata={
                "type": "image",
                "filename": Path(img_path).name,
                "extension": Path(img_path).suffix
            }
        )
        clip_docs.append(doc)
    print(f"✅ Created {len(clip_docs)} CLIP documents\n")
    
    # OCR documents (extracted text)
    print("📄 Step 2/2: Extracting text with OCR...")
    ocr_docs = extract_text_from_images(image_paths)
    
    print("\n" + "="*60)
    print(f"✅ HYBRID PROCESSING COMPLETE!")
    print(f"   📊 CLIP documents: {len(clip_docs)}")
    print(f"   📊 OCR documents: {len(ocr_docs)}")
    print("="*60 + "\n")
    
    return clip_docs, ocr_docs

def process_images_with_clip_and_captions(
    image_paths: List[str],
    caption_prompt: str | None = None
) -> tuple[List[Document], List[Document]]:
    """
    Process images for both visual similarity (CLIP) and semantic search (multimodal captions)
    
    Args:
        image_paths: List of image file paths
        caption_prompt: Optional custom prompt for caption generation
    
    Returns:
        Tuple of (clip_documents, caption_documents)
        
    Example:
        >>> paths = load_images_from_directory("./photos")
        >>> clip_docs, caption_docs = process_images_with_clip_and_captions(paths)
        >>> print(f"Visual: {len(clip_docs)}, Semantic: {len(caption_docs)}")
    """
    print("\n" + "="*60)
    print("💎 HYBRID PROCESSING: CLIP + MULTIMODAL CAPTIONS")
    print("="*60 + "\n")
    
    # CLIP documents (paths only)
    print("🖼️  Step 1/2: Creating CLIP documents...")
    clip_docs = []
    for img_path in image_paths:
        doc = Document(
            page_content=img_path,
            metadata={
                "type": "image",
                "filename": Path(img_path).name,
                "extension": Path(img_path).suffix
            }
        )
        clip_docs.append(doc)
    print(f"✅ Created {len(clip_docs)} CLIP documents\n")
    
    # Caption documents (generated descriptions)
    print("🤖 Step 2/2: Generating captions with multimodal LLM...")
    print("⏱️  This may take a while (1-3 seconds per image)...\n")
    
    caption_docs = []
    for i, img_path in enumerate(image_paths, 1):
        try:
            print(f"  [{i}/{len(image_paths)}] Processing {Path(img_path).name}...")
            caption = generate_image_caption(img_path, caption_prompt)
            
            doc = Document(
                page_content=caption,
                metadata={
                    "image_path": img_path,
                    "type": "multimodal_caption",
                    "filename": Path(img_path).name
                }
            )
            caption_docs.append(doc)
            print(f"    ✅ Caption: {caption[:60]}...\n")
            
        except Exception as e:
            print(f"    ⚠️  Caption generation failed: {e}\n")
            continue
    
    print("="*60)
    print(f"✅ HYBRID PROCESSING COMPLETE!")
    print(f"   📊 CLIP documents: {len(clip_docs)}")
    print(f"   📊 Caption documents: {len(caption_docs)}")
    print("="*60 + "\n")
    
    return clip_docs, caption_docs

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def load_images_from_directory(directory: str, extensions: List[str] | None = None) -> List[str]:
    """
    Get all image paths from a directory
    
    Args:
        directory: Directory path (absolute or relative)
        extensions: List of extensions to include (default: common image formats)
    
    Returns:
        List of absolute image file paths (sorted)
    
    Examples:
        >>> load_images_from_directory("images")  # relative path
        >>> load_images_from_directory("/home/user/images")  # absolute path
        >>> load_images_from_directory("./images")  # explicit relative path
        >>> load_images_from_directory("photos", extensions=[".jpg", ".png"])
    """
    print(f"📁 Loading images from: {directory}")
    
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    
    path = Path(directory)
    
    # Handle both absolute and relative paths
    if not path.is_absolute():
        path = path.resolve()
    
    # Check if directory exists
    if not path.exists():
        print(f"⚠️  Directory not found: {directory}")
        return []
    
    if not path.is_dir():
        print(f"⚠️  Path is not a directory: {directory}")
        return []
    
    # Find all images
    print(f"🔍 Searching for images with extensions: {', '.join(extensions)}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend([str(p) for p in path.glob(f"*{ext}")])
        image_paths.extend([str(p) for p in path.glob(f"*{ext.upper()}")])
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        print(f"⚠️  No images found in {directory}")
    else:
        print(f"✅ Found {len(image_paths)} images")
    
    return image_paths