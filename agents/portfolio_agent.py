"""
AGENT 4: Portfolio Allocation Agent.

Deterministic part : tools.allocation.compute_allocation picks the illustrative
                     equity / debt / gold split, applies the time horizon
                     guardrail, and computes the weighted assumed return.
LLM part           : explains why, how risk and horizon affect it, and what
                     risks remain. The LLM cannot change the percentages.

Runs BEFORE the Calculation Agent, because the goal calculation uses the
weighted return of this allocation as its expected return.
"""

import json

import config
from agents.llm import LLMCallError, LLMClient
from prompts.agent_prompts import PORTFOLIO_SYSTEM, PORTFOLIO_USER
from schemas.agent_outputs import PortfolioAgentOutput, PortfolioExplanationLLM
from schemas.models import GoalInput, RiskResult, UserProfile
from tools.allocation import compute_allocation
from utils.logging import SessionLog, Timer

AGENT = "portfolio_agent"


def run_portfolio_agent(
    risk: RiskResult,
    profile: UserProfile,
    goal: GoalInput,
    free_investable_surplus: float,
    llm: LLMClient | None,
    log: SessionLog | None = None,
    asset_returns: dict[str, float] | None = None,
) -> PortfolioAgentOutput:
    returns = asset_returns or config.ASSUMED_ANNUAL_RETURNS_PCT
    allocation = compute_allocation(
        risk.category, goal.horizon_years, goal.goal_type,
        profile.monthly_expenses, profile.monthly_debt, profile.emergency_fund, returns,
    )
    out = PortfolioAgentOutput(allocation=allocation)

    if llm is None:
        out.llm_error = "LLM not configured."
    else:
        facts = {
            "questionnaire_category": allocation.risk_category.value,
            "effective_category": allocation.effective_category.value,
            "guardrail_applied": allocation.guardrail_applied,
            "guardrail_reason": allocation.guardrail_reason,
            "allocation_pct": {"equity": allocation.equity_pct, "debt": allocation.debt_pct, "gold": allocation.gold_pct},
            "assumed_asset_returns_pct": returns,
            "weighted_assumed_return_pct": allocation.weighted_return_pct,
            "goal_type": goal.goal_type.value,
            "goal_horizon_years": goal.horizon_years,
            "monthly_free_investable_surplus": round(free_investable_surplus, 2),
            "emergency_reserve_gap": round(allocation.emergency_reserve_gap, 2),
            "all_allocation_tables_pct": config.ALLOCATION_TABLES,
            "assumption_label": config.ASSUMPTION_LABEL,
        }
        with Timer() as t:
            try:
                exp = llm.structured(
                    PortfolioExplanationLLM, PORTFOLIO_SYSTEM,
                    PORTFOLIO_USER.format(facts=json.dumps(facts, indent=2)),
                )
                out.explanation = exp
                out.allocation = allocation.model_copy(update={"explanation": exp.why_selected})
                out.llm_used = True
            except LLMCallError as e:
                out.llm_error = str(e)

    if log is not None:
        log.record(AGENT, "allocate_and_explain", tool="compute_allocation",
                   inputs={"risk_category": 1, "horizon_years": 1, "goal_type": 1},
                   output_summary=(f"{allocation.effective_category.value} "
                                   f"{allocation.equity_pct:.0f}/{allocation.debt_pct:.0f}/{allocation.gold_pct:.0f}, "
                                   f"guardrail={allocation.guardrail_applied}"),
                   status="ok" if out.llm_used else "fallback", error=out.llm_error,
                   duration_ms=None if llm is None else t.ms)
    return out
