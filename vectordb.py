import chromadb
from datetime import datetime, timezone

COLLECTION_NAME = "arena_embeddings"
_client = None
_collection = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.Client()
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_embeddings(
    texts: list[str],
    embeddings: list[list[float]],
    metadata: list[dict] | None = None,
) -> list[str]:
    collection = get_collection()
    ts = datetime.now(timezone.utc).isoformat()
    ids = [f"doc_{ts}_{i}" for i in range(len(texts))]
    meta = metadata or [{"source": "text", "timestamp": ts} for _ in texts]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=meta)
    return ids


def query(embedding: list[float], n_results: int = 5) -> dict:
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
