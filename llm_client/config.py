from dotenv import load_dotenv
import os
from pydantic import SecretStr

load_dotenv()

# empty string "" added to avoid intellisense yelling


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PRINT_LOG = _env_bool("PRINT_LOG")


def log_print(*args, **kwargs) -> None:
    if PRINT_LOG:
        print(*args, **kwargs)

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

# Text Embeddings settings
TEXT_EMBEDDINGS_PROVIDER = os.getenv("TEXT_EMBEDDINGS_PROVIDER", "")
TEXT_EMBEDDINGS_MODEL = os.getenv("TEXT_EMBEDDINGS_MODEL", "")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "")
DOCUMENT_PREFIX = os.getenv("DOCUMENT_PREFIX", "")

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
