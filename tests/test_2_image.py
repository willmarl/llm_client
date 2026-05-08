from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    get_image_llm,
    generate_image_caption,
    generate_image_ocr_text,
)
from config import IMAGE_PATH

print("""
Running image tests of: get_image_llm(), generate_image_caption(), generate_image_ocr_text
""")

image_location = str(IMAGE_PATH)
image_path = IMAGE_PATH

if not image_path.exists():
    print("Image not found. halting tests")
    sys.exit(1)  # or just return if inside a function

try:
    response = get_image_llm().invoke("")
    if response:
        print("get_image_llm passed ✅")
except Exception as e:
    print(f"get_image_llm failed ❌: {e}")

try:
    response = generate_image_caption(image_location)
    if response:
        print("generate_image_caption passed ✅")
except Exception as e:
    print(f"generate_image_caption failed ❌: {e}")

try:
    response = generate_image_ocr_text(image_location)
    if response:
        print("generate_image_ocr_text passed✅")
except Exception as e:
    print(f"generate_image_ocr_text failed ❌: {e}")
    print("maybe need to install tesseract? 'sudo apt-get install tesseract-ocr'")
