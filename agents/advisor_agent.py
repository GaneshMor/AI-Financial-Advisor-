"""
FINAL ADVISOR AGENT.

Builds the 10 section plan:
  1 Snapshot   2 Goal   3 Risk profile   4 Goal calculation   5 Allocation
  6 Explanation   7 Action plan   8 Risks   9 Assumptions   10 Disclaimer

Sections 1 to 5, 9 and 10 are filled DETERMINISTICALLY from the other agents.
Sections 6 to 8 (summary, explanation, action plan, risks) are written by the
LLM from a FACTS block, so every number it can use was calculated by Python.

Escalation (human advice) is decided by transparent rules, not by the LLM.
`revision_feedback` is used by the Safety Agent loop in Phase 7.
"""

import json

from agents.llm import LLMCallError, LLMClient
from agents.rag_agent import format_context
from prompts.agent_prompts import ADVISOR_SYSTEM, ADVISOR_USER, DISCLAIMER
from schemas.agent_outputs import (
    AdvisorNarrativeLLM, CalculationAgentOutput, FinalPlan, PortfolioAgentOutput, ProfileAgentOutput, RiskAgentOutput,
)
from schemas.models import RetrievedChunk
from utils.formatting import format_inr
from utils.logging import SessionLog, Timer

AGENT = "advisor_agent"


def escalation_reasons(p: ProfileAgentOutput, calc: CalculationAgentOutput) -> list[str]:
    """Rules for recommending a SEBI registered adviser. Transparent and testable."""
    reasons = []
    prof = p.profile
    if p.monthly_surplus < 0:
        reasons.append("Monthly expenses and EMIs are higher than income.")
    if prof.monthly_income > 0 and prof.monthly_debt / prof.monthly_income > 0.4:
        reasons.append("EMIs are more than 40% of income.")
    if prof.monthly_income == 0:
        reasons.append("There is no current income.")
    gap = calc.goal_plan["gap"]
    if gap["status"] == "shortfall" and not gap["gap_affordable"]:
        reasons.append("The monthly gap cannot be covered from current spare income.")
    outflow = prof.monthly_expenses + prof.monthly_debt
    if prof.dependents >= 2 and outflow > 0 and prof.emergency_fund / outflow < 3:
        reasons.append("There are dependents and the emergency fund covers under 3 months.")
    if p.goal.goal_type.value == "Retirement" and prof.age >= 50:
        reasons.append("Retirement is close, so decisions have less time to recover from mistakes.")
    return reasons


def build_sections(p: ProfileAgentOutput, r: RiskAgentOutput, pf: PortfolioAgentOutput,
                   calc: CalculationAgentOutput) -> dict:
    prof, goal, a, gp = p.profile, p.goal, pf.allocation, calc.goal_plan
    h = calc.health
    return {
        "snapshot": {
            "monthly_income": prof.monthly_income,
            "monthly_expenses": prof.monthly_expenses,
            "monthly_debt": prof.monthly_debt,
            "monthly_surplus": p.monthly_surplus,
            "current_monthly_investment": prof.monthly_investment,
            "free_investable_surplus": p.free_investable_surplus,
            "current_savings": prof.current_savings,
            "current_investments": prof.current_investments,
            "emergency_fund": prof.emergency_fund,
            "savings_rate_pct": h["savings_rate"]["value"],
            "debt_to_income_pct": h["debt_to_income"]["value"],
            "emergency_fund_months": h["emergency_fund_coverage"]["value"],
        },
        "goal": {
            "goal_type": goal.goal_type.value,
            "goal_amount_entered": goal.goal_amount,
            "amount_is_present_value": goal.amount_is_present_value,
            "horizon_years": goal.horizon_years,
            "inflation_adjusted_target": gp["target_future_value"],
            "current_goal_savings": goal.current_goal_savings,
            "priority": p.goal_priority.value if p.goal_priority else None,
        },
        "risk": {
            "score": r.risk.score,
            "category": r.risk.category.value,
            "reasoning": r.risk.reasoning,
            "key_risk_factors": r.risk.key_risk_factors,
            "factors_source": r.factors_source,
            "disclaimer": r.risk.disclaimer,
        },
        "goal_calculation": {
            "target_future_value": gp["target_future_value"],
            "fv_current_goal_savings": gp["fv_current_goal_savings"],
            "required_monthly_sip": gp["required_monthly_sip"],
            "current_monthly_sip": gp["gap"]["current_monthly_sip"],
            "monthly_gap": gp["gap"]["monthly_gap"],
            "status": gp["gap"]["status"],
            "gap_affordable_from_surplus": gp["gap"]["gap_affordable"],
            "uncovered_gap": gp["gap"]["uncovered_gap"],
            "projected_value_current_plan": gp["projected_value_current_plan"],
        },
        "allocation": {
            "questionnaire_category": a.risk_category.value,
            "effective_category": a.effective_category.value,
            "guardrail_applied": a.guardrail_applied,
            "guardrail_reason": a.guardrail_reason,
            "equity_pct": a.equity_pct, "debt_pct": a.debt_pct, "gold_pct": a.gold_pct,
            "weighted_assumed_return_pct": a.weighted_return_pct,
            "emergency_reserve_gap": a.emergency_reserve_gap,
            "label": a.label,
        },
        "assumptions": {
            "expected_annual_return_pct": calc.assumptions.expected_return_pct,
            "annual_inflation_pct": calc.assumptions.inflation_pct,
            "asset_returns_pct": calc.assumptions.asset_returns_pct,
            "horizon_years": goal.horizon_years,
            "contribution_frequency": calc.assumptions.contribution_frequency,
            "compounding": "Monthly (annual rate / 12)",
            "label": calc.assumptions.label,
        },
    }


def _facts_for_llm(sections: dict, escalation: list[str], warnings: list[str]) -> str:
    """Numbers are pre formatted in rupees so the LLM copies them instead of recomputing."""
    s = sections
    money_keys = {"monthly_income", "monthly_expenses", "monthly_debt", "monthly_surplus",
                  "current_monthly_investment", "free_investable_surplus", "current_savings",
                  "current_investments", "emergency_fund", "goal_amount_entered",
                  "inflation_adjusted_target", "current_goal_savings", "target_future_value",
                  "fv_current_goal_savings", "required_monthly_sip", "current_monthly_sip",
                  "monthly_gap", "uncovered_gap", "projected_value_current_plan", "emergency_reserve_gap"}

    def fmt(section: dict) -> dict:
        out = {}
        for k, v in section.items():
            if k in money_keys and isinstance(v, (int, float)):
                out[k] = format_inr(v)
            elif isinstance(v, float):
                out[k] = round(v, 2)
            else:
                out[k] = v
        return out

    facts = {name: fmt(sec) for name, sec in s.items()}
    facts["warnings"] = warnings
    facts["professional_advice_recommended"] = bool(escalation)
    facts["escalation_reasons"] = escalation
    return json.dumps(facts, indent=2, ensure_ascii=False)


def run_advisor_agent(
    profile_out: ProfileAgentOutput,
    risk_out: RiskAgentOutput,
    portfolio_out: PortfolioAgentOutput,
    calc_out: CalculationAgentOutput,
    context_chunks: list[RetrievedChunk],
    llm: LLMClient | None,
    log: SessionLog | None = None,
    revision_feedback: list[str] | None = None,
) -> FinalPlan:
    sections = build_sections(profile_out, risk_out, portfolio_out, calc_out)
    escalation = escalation_reasons(profile_out, calc_out)
    # business_warnings (profile agent) already cover the cash flow warnings from the health tool
    warnings = list(dict.fromkeys(profile_out.warnings + profile_out.discrepancies))

    plan = FinalPlan(
        **{k: sections[k] for k in ("snapshot", "goal", "risk", "goal_calculation", "allocation", "assumptions")},
        scenarios=calc_out.scenarios,
        warnings=warnings,
        professional_advice_recommended=bool(escalation),
        escalation_reasons=escalation,
        disclaimer=DISCLAIMER,
    )

    if llm is None:
        plan.llm_error = "LLM not configured. Numbers are complete; the written explanation is unavailable."
    else:
        with Timer() as t:
            try:
                narrative = llm.structured(
                    AdvisorNarrativeLLM, ADVISOR_SYSTEM,
                    ADVISOR_USER.format(
                        facts=_facts_for_llm(sections, escalation, warnings),
                        context=format_context(context_chunks),
                        feedback="\n".join(f"- {f}" for f in revision_feedback) if revision_feedback else "(none)",
                    ),
                )
                valid = {c.source_file for c in context_chunks}
                narrative = narrative.model_copy(
                    update={"sources_cited": [s for s in narrative.sources_cited if s in valid]}
                )
                plan.narrative = narrative
                plan.sources = narrative.sources_cited
            except LLMCallError as e:
                plan.llm_error = str(e)

    if log is not None:
        log.record(AGENT, "write_plan", inputs={"sections": 1, "context": len(context_chunks)},
                   output_summary=f"narrative={'yes' if plan.narrative else 'no'}, escalation={bool(escalation)}",
                   status="ok" if plan.narrative else "fallback", error=plan.llm_error,
                   duration_ms=None if llm is None else t.ms)
    return plan

