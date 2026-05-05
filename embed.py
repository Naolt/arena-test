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
