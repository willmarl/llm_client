from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import get_text_embeddings, generate_text_embeddings, embed_single, embed_many

print(
    """
Running text embed tests of: get_text_embeddings(), generate_text_embeddings(), embed_single(), embed_many()
"""
)

try:
    response = get_text_embeddings()
    if response:
        print("get_text_embeddings passed ✅")
except Exception as e:
    print(f"get_text_embeddings failed ❌: {e}")

try:
    response = generate_text_embeddings(["Hello"])
    if response:
        print("generate_text_embeddings passed ✅")
except Exception as e:
    print(f"generate_text_embeddings failed ❌: {e}")

try:
    response = embed_single("Hello world")
    if isinstance(response, list) and len(response) > 0:
        print("embed_single (no prefix) passed ✅")
except Exception as e:
    print(f"embed_single failed ❌: {e}")

try:
    response = embed_single("Hello world", prefix="document")
    if isinstance(response, list) and len(response) > 0:
        print("embed_single (document prefix) passed ✅")
except Exception as e:
    print(f"embed_single (document prefix) failed ❌: {e}")

try:
    response = embed_many(["Hello", "World"])
    if isinstance(response, list) and len(response) == 2:
        print("embed_many (no prefix) passed ✅")
except Exception as e:
    print(f"embed_many failed ❌: {e}")

try:
    response = embed_many(["Hello", "World"], prefix="query")
    if isinstance(response, list) and len(response) == 2:
        print("embed_many (query prefix) passed ✅")
except Exception as e:
    print(f"embed_many (query prefix) failed ❌: {e}")

try:
    response = embed_single("Hello world", prefix="custom", custom_prefix="instruct: ")
    if isinstance(response, list) and len(response) > 0:
        print("embed_single (custom prefix) passed ✅")
except Exception as e:
    print(f"embed_single (custom prefix) failed ❌: {e}")

try:
    response = embed_many(["Hello", "World"], prefix="custom", custom_prefix="passage: ")
    if isinstance(response, list) and len(response) == 2:
        print("embed_many (custom prefix) passed ✅")
except Exception as e:
    print(f"embed_many (custom prefix) failed ❌: {e}")

try:
    response = embed_single("Hello world", prefix="custom")
    print("embed_single (custom without custom_prefix) failed ❌: should have raised ValueError")
except ValueError as e:
    print("embed_single (custom without custom_prefix error handling) passed ✅")
except Exception as e:
    print(f"embed_single (custom without custom_prefix) failed ❌: {e}")
