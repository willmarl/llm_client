# Image Embedding Demo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm_client import generate_image_embeddings
from rich import print

print("Image Embedding Demo")
print("=" * 50)

# Get image from same folder
current_dir = Path(__file__).parent
img_path = current_dir / "image.jpg"

print(f"\n[Embedding: {img_path.name}]")
print("-" * 50)

# Embed the image
embedding = generate_image_embeddings(str(img_path))

print(f"✓ Successfully embedded image")
print(f"Image path: {img_path}")
print(f"Embedding type: {type(embedding)}")
print(f"Embedding dimension: {len(embedding)}")

# Display sample values
print(f"\nFirst 10 embedding values:")
for i, val in enumerate(embedding[:10]):
    print(f"  [{i}]: {val:.6f}")

# Display statistics
print(f"\nEmbedding Statistics:")
print(f"  Min value: {min(embedding):.6f}")
print(f"  Max value: {max(embedding):.6f}")
print(f"  Mean value: {sum(embedding) / len(embedding):.6f}")

print("\n" + "=" * 50)
print("ℹ️  About Image Embeddings (CLIP)")
print("-" * 50)
print("""
CLIP generates vector representations of images:
  - Used for visual similarity search
  - Can match images to text descriptions
  - Fast and efficient for large image collections
  - Each image gets a ~512D vector (depends on model)

Use cases:
  1. Find similar images by comparing embeddings
  2. Search images by text description (text-to-image search)
  3. Cluster images into groups
  4. Recommend similar visual content
""")
