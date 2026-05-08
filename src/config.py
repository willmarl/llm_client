from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

# empty string "" added to avoid intellisense yelling

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Text Embeddings settings
TEXT_EMBEDDINGS_PROVIDER = os.getenv("TEXT_EMBEDDINGS_PROVIDER", "")
TEXT_EMBEDDINGS_MODEL = os.getenv("TEXT_EMBEDDINGS_MODEL", "")

# Image model settings
IMAGE_MODEL_PROVIDER = os.getenv("IMAGE_MODEL_PROVIDER", "")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "")

# Image embeddings settings
IMAGE_EMBEDDINGS_PROVIDER = os.getenv("IMAGE_EMBEDDINGS_PROVIDER", "local")
IMAGE_EMBEDDINGS_MODEL = os.getenv("IMAGE_EMBEDDINGS_MODEL", "ViT-B/32")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "openai/clip")

# API keys & hosts
OPENAI_API_KEY = SecretStr(os.getenv("OPENAI_API_KEY", ""))
ANTHROPIC_API_KEY = SecretStr(os.getenv("ANTHROPIC_API_KEY", ""))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

###########################
# Prompts and other constants
###########################
IMAGE_CAPTION_PROMPT = os.getenv(
    "IMAGE_CAPTION_PROMPT",
    (
        "Describe this image in detail, including objects, colors, "
        "setting, mood, and any text visible. Be concise but thorough."
    ),
)

###########################
# Vector DB
###########################
VECTOR_DB_LOCATION = os.getenv("VECTOR_DB_LOCATION", "chroma_store")
