# Arena — Experiment Repo

Prototyping and experimentation for the [Arena Growth Intelligence Engine](https://www.notion.so/Position2-33e26d4c5944808cb960d26391d15a63).

---

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

---

## Experiments

### Qwen3-VL Embedding Pipeline
`notebooks/qwen_embedding_pipeline.ipynb`

Embeds marketing text using [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and stores vectors in ChromaDB. Demonstrates semantic search over campaign descriptions.

**Run on Colab** (requires GPU):
1. Open `notebooks/qwen_embedding_pipeline.ipynb` in Google Colab
2. Set runtime to GPU (Runtime → Change runtime type → T4)
3. Run all cells

**Modules:**
- `embed.py` — loads the model, exposes `embed_texts()` and `embed_query()`
- `vectordb.py` — ChromaDB setup, `store_embeddings()`, and `query()`

---

## Structure

```
arena-test/
├── embed.py
├── vectordb.py
├── notebooks/
│   └── qwen_embedding_pipeline.ipynb
├── pyproject.toml
└── README.md
```
