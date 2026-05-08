from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src import ultimate_image_extractor
from config import IMAGE_PATH

print(
    """
Running ultimate image test of: ultimate_image_extractor()
"""
)

image_location = str(IMAGE_PATH)
image_path = IMAGE_PATH

if not image_path.exists():
    print("Image not found. halting tests")
    sys.exit(1)  # or just return if inside a function


try:
    response = ultimate_image_extractor(image_location)
    if response:
        print("ultimate_image_extractor() passed ✅")
except Exception as e:
    print(f"generate_image_embeddings() failed ❌: {e}")
# response = generate_image_embeddings(image_location)
# print(response)
