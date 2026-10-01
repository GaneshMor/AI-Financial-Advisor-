"""
AGENT 5: RAG Financial Knowledge Agent.

    question -> retrieve top k chunks -> LLM answers ONLY from those chunks
             -> sources checked: any source not in the retrieved set is removed

The retriever itself is built in Phase 6 (rag/retriever.py). This agent only
needs an object with a `retrieve(query, k)` method, so it can be tested now.
"""

from typing import Protocol

from agents.llm import LLMCallError, LLMClient
from prompts.agent_prompts import RAG_SYSTEM, RAG_USER
from schemas.agent_outputs import RagAgentOutput, RagAnswerLLM
from schemas.models import AllocationResult, GoalInput, RetrievedChunk
from utils.logging import SessionLog, Timer

AGENT = "rag_agent"
NOT_FOUND = "I could not find this in the knowledge base."


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]: ...


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no knowledge context available)"
    blocks = []
    for c in chunks:
        header = f"[source: {c.source_file}] {c.title}"
        if c.source_org:
            header += f" (based on {c.source_org})"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def _retrieve(retriever, query, k, errors: list[str]) -> list[RetrievedChunk]:
    if retriever is None:
        errors.append("Knowledge base not available.")
        return []
    try:
        return list(retriever.retrieve(query, k=k))
    except Exception as e:  # index missing, embedding API error, etc.
        errors.append(f"Retrieval failed: {type(e).__name__}: {e}")
        return []


def run_rag_agent(
    question: str,
    retriever: Retriever | None,
    llm: LLMClient | None,
    log: SessionLog | None = None,
    k: int = 4,
    extra_chunks: list[RetrievedChunk] | None = None,
) -> RagAgentOutput:
    """extra_chunks: trusted context added before retrieval, e.g. the user's calculated plan."""
    out = RagAgentOutput(question=question)
    errors: list[str] = []
    with Timer() as t:
        out.retrieved = list(extra_chunks or []) + _retrieve(retriever, question, k, errors)

        if not out.retrieved:
            out.answer = NOT_FOUND
        elif llm is None:
            errors.append("LLM not configured.")
        else:
            try:
                res = llm.structured(
                    RagAnswerLLM, RAG_SYSTEM,
                    RAG_USER.format(context=format_context(out.retrieved), question=question),
                )
                valid = {c.source_file for c in out.retrieved}
                out.sources_used = [s for s in res.sources_used if s in valid]
                out.invented_sources_removed = [s for s in res.sources_used if s not in valid]
                out.answerable = res.answerable
                out.answer = res.answer if res.answerable else NOT_FOUND
            except LLMCallError as e:
                errors.append(str(e))

    out.llm_error = " | ".join(errors) or None
    if log is not None:
        log.record(AGENT, "retrieve_and_answer", tool="retriever", inputs={"question": 1},
                   output_summary=f"{len(out.retrieved)} chunks, sources={out.sources_used}",
                   status="ok" if out.answerable else "fallback", error=out.llm_error,
                   safety_flags=["invented_source_removed"] if out.invented_sources_removed else [],
                   duration_ms=t.ms)
    return out


def retrieve_plan_context(
    retriever: Retriever | None,
    goal: GoalInput,
    allocation: AllocationResult,
    k_per_query: int = 2,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Background knowledge for the Final Advisor: allocation, inflation, SIP, emergency fund."""
    queries = [
        f"{allocation.effective_category.value} asset allocation equity debt gold time horizon",
        f"inflation effect on a {goal.goal_type.value.lower()} goal",
        "how SIP and compounding work",
        "why keep an emergency fund",
    ]
    errors: list[str] = []
    seen, chunks = set(), []
    for q in queries:
        for c in _retrieve(retriever, q, k_per_query, errors):
            key = (c.source_file, c.text[:80])
            if key not in seen:
                seen.add(key)
                chunks.append(c)
        if retriever is None:
            break
    return chunks, errors
