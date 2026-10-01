"""
Convenience wrapper: run the LangGraph plan graph from already validated models.

Kept so scripts and tests can call one function. All orchestration logic now
lives in graph/plan_graph.py.
"""

from dataclasses import dataclass, field

from agents.llm import LLMClient
from agents.rag_agent import Retriever
from graph.plan_graph import SafetyFn, run_plan
from schemas.agent_outputs import (
    CalculationAgentOutput, FinalPlan, PortfolioAgentOutput, ProfileAgentOutput, RiskAgentOutput,
)
from schemas.models import GoalInput, UserProfile
from utils.logging import SessionLog


@dataclass
class PipelineResult:
    profile: ProfileAgentOutput
    risk: RiskAgentOutput
    portfolio: PortfolioAgentOutput
    calculation: CalculationAgentOutput
    plan: FinalPlan
    rag_errors: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    safety_status: str = "not_run"
    context_chunks: list = field(default_factory=list)


def run_plan_pipeline(
    profile: UserProfile,
    goal: GoalInput,
    risk_answers: list[int],
    llm: LLMClient | None,
    retriever: Retriever | None = None,
    log: SessionLog | None = None,
    expected_return_pct: float | None = None,
    inflation_pct: float | None = None,
    safety_fn: SafetyFn | None = None,
) -> PipelineResult:
    res = run_plan(profile.model_dump(), goal.model_dump(), risk_answers, llm, retriever, safety_fn, log,
                   expected_return_pct, inflation_pct)
    if not res.ok:
        raise RuntimeError("Plan generation failed: " + "; ".join(res.errors))
    return PipelineResult(res.profile, res.risk, res.portfolio, res.calculation, res.plan,
                          res.errors, res.trace, res.safety_status, res.context_chunks)
