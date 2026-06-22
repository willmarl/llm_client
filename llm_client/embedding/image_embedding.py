from typing import List
from ..config import (
    REPLICATE_MODEL,
    IMAGE_EMBEDDINGS_PROVIDER,
    IMAGE_EMBEDDINGS_MODEL,
)
from pathlib import Path

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
    import warnings

    try:
        import torch
        import clip
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Local CLIP image embeddings require torch, CLIP, and Pillow. Install llm_client[full], then install torch/CLIP for your platform."
        ) from exc

    # Suppress ROCm/HIP experimental feature warnings on AMD GPUs (e.g. Navi31/7900 XTX)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            backend = f"ROCm {hip_version.split('-')[0]} (AMD)"
        else:
            backend = "CUDA (NVIDIA)"
        print(f"  🔄 Processing with local CLIP on GPU: {gpu_name} via {backend}...")
    else:
        print(f"  🔄 Processing with local CLIP on CPU...")

    model, preprocess = clip.load(IMAGE_EMBEDDINGS_MODEL, device=device)
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)  # type: ignore

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

    try:
        import pybase64
        import replicate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Replicate image embeddings require optional dependencies. Install with: pip install 'llm_client[vision]'"
        ) from exc

    base64_image = pybase64.b64encode(Path(path).read_bytes()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{base64_image}"

    output = replicate.run(REPLICATE_MODEL, input={"image": data_uri})

    result: dict = output  # type: ignore
    return result["embedding"]


def generate_image_embeddings(path: str) -> List[float]:
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
        raise ValueError(
            f"❌ Unknown image embeddings provider: {IMAGE_EMBEDDINGS_PROVIDER}"
        )
