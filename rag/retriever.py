"""
Retrievers used by the RAG agent. Both return schemas.models.RetrievedChunk.

FaissRetriever   : semantic search. Embeds the question with the same OpenAI
                   model used at ingest, cosine similarity via FAISS, keeps
                   hits with score >= config.RAG_MIN_SCORE.
KeywordRetriever : BM25 keyword search over the same chunks. No API key or
                   index needed. Used as a FALLBACK, and labelled as such.

get_retriever() picks FAISS when the index exists and the API key is set,
otherwise the keyword fallback, and records why in `.method` / `.warning`.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

import config
from rag.documents import Chunk, kb_fingerprint, load_chunks
from schemas.models import RetrievedChunk


def _to_retrieved(c: dict | Chunk, score: float) -> RetrievedChunk:
    d = c if isinstance(c, dict) else c.__dict__
    return RetrievedChunk(text=d["text"], source_file=d["source_file"], title=f"{d['title']} > {d['section']}",
                          source_org=d.get("source_org"), source_url=d.get("source_url"), score=round(float(score), 4))


# ---------------------------------------------------------------------------
# FAISS (semantic)
# ---------------------------------------------------------------------------
class FaissRetriever:
    method = "faiss"

    def __init__(self, embeddings, index_dir: Path | None = None, min_score: float | None = None,
                 kb_dir: Path | None = None):
        import faiss

        index_dir = Path(index_dir or config.RAG_INDEX_DIR)
        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index in {index_dir}. Run: python -m rag.ingest")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.embeddings = embeddings
        self.min_score = config.RAG_MIN_SCORE if min_score is None else min_score
        self.warning = None
        if self.manifest.get("kb_fingerprint") != kb_fingerprint(kb_dir):
            self.warning = "Knowledge base files changed after the index was built. Run: python -m rag.ingest"

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[RetrievedChunk]:
        vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        scores, ids = self.index.search(vec, min(k, len(self.chunks)))
        return [_to_retrieved(self.chunks[i], s) for s, i in zip(scores[0], ids[0])
                if i >= 0 and s >= self.min_score]


# ---------------------------------------------------------------------------
# BM25 (keyword fallback)
# ---------------------------------------------------------------------------
STOPWORDS = set("""
a an the and or of to in on for is are was were be been it its this that these those with as at by from
what why how when which who do does did can could should would will my your our i you we they me us
about into than then there their them so if not no yes any all more most much many also just
""".split())


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # very light stemming: funds -> fund
        out.append(w)
    return out


class KeywordRetriever:
    method = "keyword (BM25 fallback)"

    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks or load_chunks()
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]
        self.warning = "Using keyword search. Build the FAISS index for semantic search: python -m rag.ingest"

    def _score(self, q_tokens: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in q_tokens:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def retrieve(self, query: str, k: int = config.RAG_TOP_K) -> list[RetrievedChunk]:
        q = tokenize(query)
        if not q:
            return []
        scored = sorted(((self._score(q, i), i) for i in range(len(self.chunks))), reverse=True)
        return [_to_retrieved(self.chunks[i], s) for s, i in scored[:k] if s > 0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_retriever(embeddings=None, index_dir: Path | None = None):
    """FAISS if possible, else keyword fallback. Returns None only if the knowledge base is missing."""
    index_dir = Path(index_dir or config.RAG_INDEX_DIR)
    reason = None
    if (index_dir / "index.faiss").exists():
        try:
            if embeddings is None:
                from rag.ingest import get_embeddings
                embeddings = get_embeddings()
            r = FaissRetriever(embeddings, index_dir)
            if r.manifest.get("embedding_model") != config.EMBEDDING_MODEL and embeddings is not None:
                reason = (f"Index was built with {r.manifest.get('embedding_model')} but EMBEDDING_MODEL is "
                          f"{config.EMBEDDING_MODEL}. Re-run: python -m rag.ingest")
            else:
                return r
        except Exception as e:  # noqa: BLE001 - missing key, corrupt index, etc.
            reason = f"FAISS unavailable ({type(e).__name__}: {e})"
    try:
        kr = KeywordRetriever()
    except FileNotFoundError:
        return None
    if reason:
        kr.warning = reason + ". Using keyword search instead."
    return kr
