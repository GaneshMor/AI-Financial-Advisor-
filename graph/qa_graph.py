"""
LangGraph orchestration: GRAPH 2, AI Advisor Q&A.

    START -> router (LLM) -> calculation  (tool calling agent)  -> safety -> END
                          -> knowledge    (RAG agent + your plan) -> safety -> END
                          -> out_of_scope (fixed polite refusal)  -> END

* No LLM configured -> a clear message; nothing is faked.
* Router failure    -> error returned; the user can retry.
* Safety: `safety_fn(text, state) -> SafetyReport` is plugged in by Phase 7.
  An answer with a HIGH severity flag is withheld, with the reason shown.
"""

import json
import operator
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.calculation_agent import answer_calculation_question
from agents.llm import LLMCallError, LLMClient
from agents.rag_agent import Retriever, run_rag_agent
from prompts.agent_prompts import OUT_OF_SCOPE_ANSWER, ROUTER_SYSTEM
from schemas.agent_outputs import FinalPlan, QAResult, RouteLLM, ToolCallRecord
from schemas.models import RetrievedChunk, SafetyReport, Severity
from utils.formatting import format_inr
from utils.logging import SessionLog

NO_LLM_ANSWER = (
    "The AI Advisor needs an OpenAI API key (see .env). Your plan, calculations and "
    "What If page still work without it."
)
PLAN_SOURCE = "your_plan"

QASafetyFn = Callable[[str, dict], SafetyReport]


class QAState(TypedDict, total=False):
    question: str
    plan_context: dict
    plan_chunk: RetrievedChunk | None
    context_chunks: list[RetrievedChunk]
    route: str
    route_reason: str
    answer: str | None
    tool_calls: list[ToolCallRecord]
    sources_used: list[str]
    invented_sources_removed: list[str]
    safety_flags: list[str]
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]


def plan_facts_chunk(plan: FinalPlan) -> RetrievedChunk:
    """The user's calculated plan as a trusted context passage for knowledge questions."""
    g, gc, a, r, asm = plan.goal, plan.goal_calculation, plan.allocation, plan.risk, plan.assumptions
    lines = [
        f"Goal: {g['goal_type']} in {g['horizon_years']} years. Inflation adjusted target {format_inr(gc['target_future_value'])}.",
        f"Risk questionnaire: score {r['score']} ({r['category']}). Bands: 7-11 Conservative, 12-16 Moderate, 17-21 Aggressive.",
        f"Effective allocation category: {a['effective_category']}."
        + (f" Guardrail applied: {a['guardrail_reason']}" if a["guardrail_applied"] else ""),
        f"Illustrative allocation: equity {a['equity_pct']:.0f}%, debt {a['debt_pct']:.0f}%, gold {a['gold_pct']:.0f}%.",
        f"Assumed returns (illustrative): {asm['asset_returns_pct']}; weighted {a['weighted_assumed_return_pct']}%. "
        f"Inflation assumption {asm['annual_inflation_pct']}%.",
        f"Required monthly SIP {format_inr(gc['required_monthly_sip'])}; current SIP {format_inr(gc['current_monthly_sip'])}; "
        f"status {gc['status']}.",
    ]
    return RetrievedChunk(text="\n".join(lines), source_file=PLAN_SOURCE,
                          title="Your plan (calculated by the app)", score=1.0)


def build_qa_graph(
    llm: LLMClient | None,
    retriever: Retriever | None = None,
    safety_fn: QASafetyFn | None = None,
    log: SessionLog | None = None,
):
    def router(s: QAState) -> dict:
        if llm is None:
            return {"route": "no_llm", "answer": NO_LLM_ANSWER, "trace": ["router (no LLM)"]}
        try:
            r = llm.structured(RouteLLM, ROUTER_SYSTEM, f"QUESTION: {s['question']}")
        except LLMCallError as e:
            return {"route": "error", "errors": [f"router: {e}"], "trace": ["router (failed)"]}
        if log:
            log.record("router_agent", "route", inputs={"question": 1}, output_summary=r.route)
        return {"route": r.route, "route_reason": r.reason, "trace": [f"router -> {r.route}"]}

    def calculation(s: QAState) -> dict:
        out = answer_calculation_question(s["question"], s.get("plan_context"), llm, log)
        return {"answer": out.answer, "tool_calls": out.tool_calls,
                "errors": [out.llm_error] if out.llm_error else [], "trace": ["calculation_agent"]}

    def knowledge(s: QAState) -> dict:
        extra = [s["plan_chunk"]] if s.get("plan_chunk") else []
        out = run_rag_agent(s["question"], retriever, llm, log, extra_chunks=extra)
        return {"answer": out.answer, "sources_used": out.sources_used, "context_chunks": out.retrieved,
                "invented_sources_removed": out.invented_sources_removed,
                "errors": [out.llm_error] if out.llm_error else [], "trace": ["rag_agent"]}

    def out_of_scope(s: QAState) -> dict:
        return {"answer": OUT_OF_SCOPE_ANSWER, "trace": ["out_of_scope"]}

    def safety(s: QAState) -> dict:
        if not s.get("answer"):
            return {"trace": ["safety (nothing to check)"]}
        try:
            report = safety_fn(s["answer"], dict(s))
        except Exception as e:  # noqa: BLE001
            return {"errors": [f"safety: {type(e).__name__}: {e}"], "trace": ["safety (failed)"]}
        flags = [f"{f.severity.value}:{f.category.value}" for f in report.flags]
        high = [f for f in report.flags if f.severity == Severity.HIGH]
        update = {"safety_flags": flags, "trace": ["safety"]}
        if high:
            update["answer"] = ("This answer was withheld because it failed a safety check: "
                                + "; ".join(f.text for f in high))
        return update

    def route_edge(s: QAState) -> str:
        return s["route"] if s.get("route") in {"calculation", "knowledge", "out_of_scope"} else END

    after_answer = "safety" if safety_fn else END

    g = StateGraph(QAState)
    g.add_node("router", router)
    g.add_node("calculation", calculation)
    g.add_node("knowledge", knowledge)
    g.add_node("out_of_scope", out_of_scope)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_edge, ["calculation", "knowledge", "out_of_scope", END])
    g.add_edge("out_of_scope", END)
    if safety_fn:
        g.add_node("safety", safety)
        g.add_edge("safety", END)
    g.add_edge("calculation", after_answer)
    g.add_edge("knowledge", after_answer)
    return g.compile()


def ask(
    question: str,
    llm: LLMClient | None,
    retriever: Retriever | None = None,
    plan: FinalPlan | None = None,
    plan_context: dict | None = None,
    safety_fn: QASafetyFn | None = None,
    log: SessionLog | None = None,
) -> QAResult:
    question = (question or "").strip()
    if not question:
        return QAResult(question="", errors=["Please type a question."])
    graph = build_qa_graph(llm, retriever, safety_fn, log)
    s = graph.invoke({
        "question": question[:1000],
        "plan_context": plan_context or {},
        "plan_chunk": plan_facts_chunk(plan) if plan else None,
        "errors": [], "trace": [],
    })
    return QAResult(
        question=question,
        route=s.get("route"),
        route_reason=s.get("route_reason"),
        answer=s.get("answer"),
        tool_calls=s.get("tool_calls", []),
        sources_used=s.get("sources_used", []),
        invented_sources_removed=s.get("invented_sources_removed", []),
        safety_flags=s.get("safety_flags", []),
        errors=[e for e in s.get("errors", []) if e],
        trace=s.get("trace", []),
    )


def export_json(result: QAResult) -> str:
    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
