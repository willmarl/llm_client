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
    log_print,
)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# ============================================
# LLM FUNCTIONS
# ============================================


def get_llm(temperature: float = 0.7):
    """
    Get the LLM based on provider and model from config

    Args:
        temperature (float): Sampling temperature (0-2). Higher = more creative, Lower = more deterministic

    Returns:
        LLM instance (ChatOpenAI, ChatAnthropic, or ChatOllama)
    """
    log_print(f"🤖 Initializing LLM: {LLM_PROVIDER} ({LLM_MODEL}) - Temperature: {temperature}")

    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=temperature)
    elif LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model_name=LLM_MODEL, api_key=ANTHROPIC_API_KEY, temperature=temperature, timeout=None, stop=None
        )
    elif LLM_PROVIDER == "ollama":
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=temperature)
    else:
        raise ValueError(f"❌ Unknown provider: {LLM_PROVIDER}")


def run_llm(prompt: str):
    """
    Run the LLM with the given prompt and return the response text.

    Args:
        prompt (str): The input prompt for the LLM.

    Returns:
        str: The response from the LLM.
    """
    llm = get_llm()
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content
