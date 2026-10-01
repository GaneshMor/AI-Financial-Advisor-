"""
Export both LangGraph graphs as Mermaid diagrams (for the report and viva).

    python -m scripts.export_graphs

Writes docs/plan_graph.mmd and docs/qa_graph.mmd. Paste them into
https://mermaid.live to get an image, or view them on GitHub.
The graphs are exported WITH a safety step so the full design is shown.
"""

from pathlib import Path

from graph.plan_graph import build_plan_graph
from graph.qa_graph import build_qa_graph
from schemas.models import SafetyReport

DOCS = Path(__file__).resolve().parent.parent / "docs"


def _dummy_safety(*_):
    return SafetyReport()


def main():
    DOCS.mkdir(exist_ok=True)
    plan = build_plan_graph(None, None, safety_fn=_dummy_safety).get_graph().draw_mermaid()
    qa = build_qa_graph(None, None, safety_fn=_dummy_safety).get_graph().draw_mermaid()
    (DOCS / "plan_graph.mmd").write_text(plan, encoding="utf-8")
    (DOCS / "qa_graph.mmd").write_text(qa, encoding="utf-8")
    print("Wrote docs/plan_graph.mmd and docs/qa_graph.mmd")


if __name__ == "__main__":
    main()
