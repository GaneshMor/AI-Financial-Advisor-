"""
Build the FAISS vector index from the knowledge base.

    python -m rag.ingest

Steps: load markdown -> chunk -> embed each chunk (OpenAI embeddings) ->
normalise vectors -> FAISS inner product index (= cosine similarity) ->
save index.faiss, chunks.json and manifest.json in rag/index/.

Re-run this whenever you edit a knowledge base file. The retriever warns if
the index is older than the files.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from rag.documents import load_chunks, kb_fingerprint


def get_embeddings():
    """OpenAI embeddings. Raises RuntimeError if the API key is missing."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. It is needed to embed the knowledge base.")
    from langchain_openai import OpenAIEmbeddings

    # Chunks are short (< 1,000 characters), so token length checks are not needed.
    # Turning them off also avoids a tokenizer download on first run.
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY,
                            base_url=config.LLM_BASE_URL, check_embedding_ctx_length=False)


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def build_index(embeddings=None, kb_dir: Path | None = None, index_dir: Path | None = None,
                model_name: str | None = None) -> dict:
    import faiss

    kb_dir = Path(kb_dir or config.KNOWLEDGE_BASE_DIR)
    index_dir = Path(index_dir or config.RAG_INDEX_DIR)
    embeddings = embeddings or get_embeddings()

    chunks = load_chunks(kb_dir)
    vectors = normalise(np.array(embeddings.embed_documents([c.text for c in chunks]), dtype="float32"))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    manifest = {
        "embedding_model": model_name or config.EMBEDDING_MODEL,
        "dimension": int(vectors.shape[1]),
        "chunks": len(chunks),
        "files": sorted({c.source_file for c in chunks}),
        "kb_fingerprint": kb_fingerprint(kb_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    m = build_index()
    print(f"Indexed {m['chunks']} chunks from {len(m['files'])} files with {m['embedding_model']} "
          f"(dimension {m['dimension']}) -> {config.RAG_INDEX_DIR}")


if __name__ == "__main__":
    main()
