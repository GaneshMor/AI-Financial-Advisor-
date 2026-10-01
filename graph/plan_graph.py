"""
LangGraph orchestration: GRAPH 1, Plan Generation.

    START -> validate -> profile -> risk -> portfolio -> calculation
          -> knowledge -> advisor -> safety -> (revise? back to advisor) -> finalize -> END

* Every node is wrapped by `_guard`: an unexpected exception is recorded in
  state["errors"] and the graph stops cleanly instead of crashing the app.
* A failed validation or calculation is FATAL (no plan without numbers).
  LLM failures are NOT fatal: agents fall back and the plan still completes.
* Safety: `safety_fn(plan, state) -> SafetyReport` is plugged in by Phase 7.
  If any HIGH severity flag is found, the advisor rewrites the narrative with
  the flags as feedback, at most MAX_REVISIONS times.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.advisor_agent import run_advisor_agent
from agents.calculation_agent import run_calculation_agent
from agents.llm import LLMClient
from agents.portfolio_agent import run_portfolio_agent
from agents.profile_agent import run_profile_agent
from agents.rag_agent import Retriever, retrieve_plan_context
from agents.risk_agent import run_risk_agent
from schemas.agent_outputs import (
    CalculationAgentOutput, FinalPlan, PortfolioAgentOutput, ProfileAgentOutput, RiskAgentOutput,
)
from schemas.models import GoalInput, RetrievedChunk, SafetyReport, Severity, UserProfile
from utils.logging import SessionLog
from utils.validation import validate_inputs

MAX_REVISIONS = 2

SafetyFn = Callable[[FinalPlan, dict], SafetyReport]


class PlanState(TypedDict, total=False):
    # inputs
    profile_data: dict
    goal_data: dict
    risk_answers: list[int]
    expected_return_pct: float | None
    inflation_pct: float | None
    # validated inputs
    profile: UserProfile
    goal: GoalInput
    # agent outputs
    profile_out: ProfileAgentOutput
    risk_out: RiskAgentOutput
    portfolio_out: PortfolioAgentOutput
    calc_out: CalculationAgentOutput
    context_chunks: list[RetrievedChunk]
    plan: FinalPlan
    # safety loop
    safety_report: SafetyReport | None
    revision_count: int
    revision_feedback: list[str]
    safety_status: str      # not_run | passed | passed_structure_only | failed_after_revisions | error
    # control
    fatal: bool
    errors: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]


def _guard(name: str, fn: Callable[[PlanState], dict], fatal_on_error: bool = True):
    """Run a node; on an unexpected exception record it instead of crashing."""
    def node(state: PlanState) -> dict:
        try:
            update = fn(state) or {}
            update.setdefault("trace", [name])
            return update
        except Exception as e:  # noqa: BLE001 - graph must never crash the UI
            return {"errors": [f"{name}: {type(e).__name__}: {e}"], "fatal": fatal_on_error, "trace": [f"{name} (failed)"]}
    return node


def _high_flags(report: SafetyReport | None) -> list:
    return [f for f in (report.flags if report else []) if f.severity == Severity.HIGH]


def build_plan_graph(
    llm: LLMClient | None,
    retriever: Retriever | None = None,
    safety_fn: SafetyFn | None = None,
    log: SessionLog | None = None,
):
    # ------------------------------------------------------------------ nodes
    def validate(s: PlanState) -> dict:
        rep = validate_inputs(s["profile_data"], s["goal_data"])
        if not rep.ok:
            return {"errors": rep.errors, "fatal": True, "trace": ["validate (invalid input)"]}
        if log:
            log.record("orchestrator", "validate", inputs=s["profile_data"] | s["goal_data"],
                       output_summary=f"{len(rep.warnings)} warnings")
        return {"profile": rep.profile, "goal": rep.goal, "fatal": False}

    def profile(s):
        return {"profile_out": run_profile_agent(s["profile"], s["goal"], llm, log)}

    def risk(s):
        return {"risk_out": run_risk_agent(s["risk_answers"], llm, s["profile"], s["goal"], log)}

    def portfolio(s):
        return {"portfolio_out": run_portfolio_agent(
            s["risk_out"].risk, s["profile"], s["goal"], s["profile_out"].free_investable_surplus, llm, log)}

    def calculation(s):
        return {"calc_out": run_calculation_agent(
            s["profile"], s["goal"], s["portfolio_out"].allocation, log,
            s.get("expected_return_pct"), s.get("inflation_pct"))}

    def knowledge(s):
        chunks, errs = retrieve_plan_context(retriever, s["goal"], s["portfolio_out"].allocation)
        return {"context_chunks": chunks, "errors": errs}

    def advisor(s):
        plan = run_advisor_agent(
            s["profile_out"], s["risk_out"], s["portfolio_out"], s["calc_out"],
            s.get("context_chunks", []), llm, log, s.get("revision_feedback") or None)
        return {"plan": plan}

    def safety(s):
        report = safety_fn(s["plan"], dict(s))
        high = _high_flags(report)
        if log:
            log.record("safety_agent", "check_plan", inputs={"plan": 1},
                       output_summary=f"{len(report.flags)} flags, {len(high)} high",
                       status="ok" if not high else "fallback",
                       safety_flags=[f.category.value for f in report.flags])
        return {"safety_report": report}

    def revise(s):
        feedback = [
            f"[{f.category.value}] {f.text}" + (f" -> {f.suggestion}" if f.suggestion else "")
            for f in _high_flags(s.get("safety_report"))
        ]
        return {"revision_feedback": feedback, "revision_count": s.get("revision_count", 0) + 1}

    def finalize(s):
        if safety_fn is None:
            status = "not_run"
        elif s.get("safety_report") is None:
            status = "error"
        elif _high_flags(s["safety_report"]):
            status = "failed_after_revisions"
        elif s["plan"].narrative is None:
            status = "passed_structure_only"  # no AI text to check; numbers and allocation checked
        else:
            status = "passed"
        update = {"safety_status": status}
        if status == "failed_after_revisions" and s["plan"].narrative is not None:
            # Never show AI text that still fails a HIGH severity check. Numbers stay.
            problems = "; ".join(f"{f.category.value}: {f.text[:80]}" for f in _high_flags(s["safety_report"]))
            update["plan"] = s["plan"].model_copy(update={
                "narrative": None, "sources": [],
                "llm_error": f"AI explanation withheld after {MAX_REVISIONS} failed safety revisions ({problems}).",
            })
        return update

    # ------------------------------------------------------------------ graph
    g = StateGraph(PlanState)
    g.add_node("validate", _guard("validate", validate))
    g.add_node("profile", _guard("profile", profile))
    g.add_node("risk", _guard("risk", risk))
    g.add_node("portfolio", _guard("portfolio", portfolio))
    g.add_node("calculation", _guard("calculation", calculation))
    g.add_node("knowledge", _guard("knowledge", knowledge, fatal_on_error=False))
    g.add_node("advisor", _guard("advisor", advisor))
    g.add_node("finalize", _guard("finalize", finalize, fatal_on_error=False))

    def next_or_end(nxt: str):
        return lambda s: END if s.get("fatal") else nxt

    chain = ["validate", "profile", "risk", "portfolio", "calculation", "knowledge", "advisor"]
    g.add_edge(START, "validate")
    for a, b in zip(chain, chain[1:]):
        g.add_conditional_edges(a, next_or_end(b), [b, END])

    if safety_fn is None:
        g.add_conditional_edges("advisor", next_or_end("finalize"), ["finalize", END])
    else:
        g.add_node("safety", _guard("safety", safety, fatal_on_error=False))
        g.add_node("revise", _guard("revise", revise))
        g.add_conditional_edges("advisor", next_or_end("safety"), ["safety", END])

        def after_safety(s):
            if _high_flags(s.get("safety_report")) and s.get("revision_count", 0) < MAX_REVISIONS:
                return "revise"
            return "finalize"

        g.add_conditional_edges("safety", after_safety, ["revise", "finalize"])
        g.add_edge("revise", "advisor")

    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------- API
@dataclass
class PlanRunResult:
    ok: bool
    errors: list[str]
    trace: list[str]
    plan: FinalPlan | None = None
    profile: ProfileAgentOutput | None = None
    risk: RiskAgentOutput | None = None
    portfolio: PortfolioAgentOutput | None = None
    calculation: CalculationAgentOutput | None = None
    safety_report: SafetyReport | None = None
    safety_status: str = "not_run"
    revision_count: int = 0
    context_chunks: list[RetrievedChunk] = field(default_factory=list)


def run_plan(
    profile_data: dict,
    goal_data: dict,
    risk_answers: list[int],
    llm: LLMClient | None,
    retriever: Retriever | None = None,
    safety_fn: SafetyFn | None = None,
    log: SessionLog | None = None,
    expected_return_pct: float | None = None,
    inflation_pct: float | None = None,
) -> PlanRunResult:
    graph = build_plan_graph(llm, retriever, safety_fn, log)
    s = graph.invoke({
        "profile_data": profile_data, "goal_data": goal_data, "risk_answers": risk_answers,
        "expected_return_pct": expected_return_pct, "inflation_pct": inflation_pct,
        "revision_count": 0, "errors": [], "trace": [],
    })
    return PlanRunResult(
        ok=not s.get("fatal", False) and "plan" in s,
        errors=s.get("errors", []),
        trace=s.get("trace", []),
        plan=s.get("plan"),
        profile=s.get("profile_out"),
        risk=s.get("risk_out"),
        portfolio=s.get("portfolio_out"),
        calculation=s.get("calc_out"),
        safety_report=s.get("safety_report"),
        safety_status=s.get("safety_status", "not_run"),
        revision_count=s.get("revision_count", 0),
        context_chunks=s.get("context_chunks", []),
    )
