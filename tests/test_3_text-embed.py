from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src import get_text_embeddings, generate_text_embeddings
from config import IMAGE_PATH

print(
    """
Running text embed tests of: get_text_embeddings(), generate_text_embeddings()
"""
)

image_location = str(IMAGE_PATH)
image_path = IMAGE_PATH

if not image_path.exists():
    print("Image not found. halting tests")
    sys.exit(1)  # or just return if inside a function

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
