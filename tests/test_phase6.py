"""
Phase 6 tests: knowledge base files, chunking, keyword retrieval quality on the
labelled cases, FAISS index mechanics, retriever selection, and the RAG agent
running on the real knowledge base.

FAISS tests use DeterministicFakeEmbedding (same text -> same vector) so they
check index MECHANICS without an API key. Semantic quality of real OpenAI
embeddings is measured in Phase 9.
"""

import csv
import json
import shutil
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import config
from agents.rag_agent import run_rag_agent
from rag.documents import KBDocument, chunk_document, load_chunks, parse_document
from rag.ingest import build_index
from rag.retriever import FaissRetriever, KeywordRetriever, get_retriever, tokenize
from schemas.agent_outputs import RagAnswerLLM
from tests.fakes import FakeLLM

ROOT = Path(__file__).resolve().parent.parent
KB = config.KNOWLEDGE_BASE_DIR
with (ROOT / "data" / "test_cases.csv").open(encoding="utf-8") as f:
    RAG_CASES = [(json.loads(c["input_json"])["question"], json.loads(c["expected_json"])["expected_source"])
                 for c in csv.DictReader(f) if c["category"] == "rag_retrieval"]


# ---------------------------------------------------------------------------
# Knowledge base files
# ---------------------------------------------------------------------------
def test_all_planned_files_exist():
    assert sorted(p.name for p in KB.glob("*.md")) == sorted(config.KNOWLEDGE_BASE_FILES)


@pytest.mark.parametrize("name", config.KNOWLEDGE_BASE_FILES)
def test_every_file_has_real_sources(name):
    doc = parse_document(KB / name)
    assert doc.title and doc.source_org and doc.last_checked
    assert doc.sources and all(url.startswith("https://") for _, url in doc.sources)
    allowed = ("sebi.gov.in", "mutualfundssahihai.com", "ncfe.org.in")
    assert all(any(a in url for a in allowed) for _, url in doc.sources), "only official/industry-body sources"
    assert len(doc.body.split()) > 120


def test_parse_rejects_missing_front_matter(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("# No header\ntext", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_document(p)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunks_are_self_describing_and_unique():
    chunks = load_chunks()
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.source_file for c in chunks} == set(config.KNOWLEDGE_BASE_FILES)
    for c in chunks:
        assert c.text.startswith(f"{c.title} > {c.section}")
        assert len(c.text) <= config.RAG_CHUNK_CHARS + 120


def test_long_section_is_split_with_overlap():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(6)]
    doc = KBDocument("t.md", "T", "Org", [("n", "https://x")], "", "## Long\n" + "\n\n".join(paras))
    chunks = chunk_document(doc, limit=500)
    assert len(chunks) > 1
    last_para_of_first = chunks[0].text.split("\n\n")[-1]
    assert last_para_of_first in chunks[1].text  # overlap


# ---------------------------------------------------------------------------
# Keyword retrieval (fallback) on the labelled questions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question, expected", RAG_CASES)
def test_keyword_retriever_finds_expected_source_in_top3(question, expected):
    hits = KeywordRetriever().retrieve(question, k=3)
    assert expected in [h.source_file for h in hits]


def test_keyword_edge_cases():
    r = KeywordRetriever()
    assert r.retrieve("the and of") == []
    assert r.retrieve("zzqx unrelatedword") == []
    assert tokenize("Mutual Funds") == ["mutual", "fund"]


# ---------------------------------------------------------------------------
# FAISS index mechanics
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_index(tmp_path):
    kb = tmp_path / "kb"
    shutil.copytree(KB, kb)
    emb = DeterministicFakeEmbedding(size=64)
    manifest = build_index(emb, kb_dir=kb, index_dir=tmp_path / "index", model_name="fake")
    return emb, kb, tmp_path / "index", manifest


def test_build_index_manifest(fake_index):
    _, _, index_dir, m = fake_index
    assert m["chunks"] == len(load_chunks()) and m["dimension"] == 64
    assert len(m["files"]) == 13
    assert (index_dir / "index.faiss").exists() and (index_dir / "chunks.json").exists()


def test_faiss_exact_match_and_threshold(fake_index):
    emb, kb, index_dir, _ = fake_index
    r = FaissRetriever(emb, index_dir, min_score=0.99, kb_dir=kb)
    target = load_chunks(kb)[5]
    hits = r.retrieve(target.text, k=3)
    assert hits and hits[0].source_file == target.source_file and hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert len(hits) == 1  # other chunks fall below the 0.99 threshold
    assert r.warning is None


def test_faiss_detects_stale_index(fake_index):
    emb, kb, index_dir, _ = fake_index
    (kb / "sip.md").write_text((kb / "sip.md").read_text(encoding="utf-8") + "\nNew line.", encoding="utf-8")
    assert "changed" in FaissRetriever(emb, index_dir, kb_dir=kb).warning


def test_get_retriever_falls_back_without_index(tmp_path):
    r = get_retriever(index_dir=tmp_path / "missing")
    assert isinstance(r, KeywordRetriever)


def test_get_retriever_falls_back_when_embeddings_fail(fake_index, monkeypatch):
    _, _, index_dir, _ = fake_index
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    r = get_retriever(index_dir=index_dir)
    assert isinstance(r, KeywordRetriever) and "FAISS unavailable" in r.warning


# ---------------------------------------------------------------------------
# RAG agent on the real knowledge base
# ---------------------------------------------------------------------------
def test_rag_agent_with_real_knowledge_base():
    answer = RagAnswerLLM(answer="Inflation reduces purchasing power (source: inflation.md)",
                          sources_used=["inflation.md", "imf_report.pdf"], answerable=True)
    llm = FakeLLM({"RagAnswerLLM": answer})
    out = run_rag_agent("Why does inflation matter for long term goals?", KeywordRetriever(), llm)
    assert out.sources_used == ["inflation.md"] and out.invented_sources_removed == ["imf_report.pdf"]
    prompt = llm.calls[0][2]
    assert "[source: inflation.md]" in prompt and "(1 + inflation rate)" in prompt
