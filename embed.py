from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], instruction: str = "Represent the user's input.") -> list[list[float]]:
    embeddings = _get_model().encode(texts, prompt=instruction, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    embeddings = _get_model().encode(
        [query],
        prompt="Retrieve relevant documents for the query.",
        convert_to_numpy=True,
    )
    return embeddings[0].tolist()


def embed_images(images: list[Image.Image], batch_size: int = 1) -> list[list[float]]:
    model = _get_model()
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        all_embeddings.extend(model.encode(batch, convert_to_numpy=True).tolist())
    return all_embeddings


def embed_multimodal(items: list[str | dict | Image.Image], batch_size: int = 1) -> list[list[float]]:
    """
    Accepts a mixed list. Each item can be:
      - str: plain text or image URL/file path
      - PIL Image
      - dict with "text" and/or "image" keys, e.g. {"text": "...", "image": "path/or/url"}
    """
    model = _get_model()
    all_embeddings = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        all_embeddings.extend(model.encode(batch, convert_to_numpy=True).tolist())
    return all_embeddings
