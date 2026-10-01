"""
Inspect retrieval from the command line (no LLM needed).

    python -m rag.query "Why does inflation matter?"

Shows which retriever is active, and the top chunks with scores and sources.
"""

import sys

from rag.retriever import get_retriever


def main(question: str) -> None:
    r = get_retriever()
    if r is None:
        raise SystemExit("Knowledge base not found in rag/knowledge_base/")
    print(f"Retriever: {r.method}")
    if r.warning:
        print(f"Note: {r.warning}")
    hits = r.retrieve(question)
    if not hits:
        print("No relevant chunks found.")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h.score}  source={h.source_file}  ({h.source_org})")
        print(f"    {h.title}")
        print("    " + h.text.split("\n", 1)[-1][:300].replace("\n", " "))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "What is a SIP?")
