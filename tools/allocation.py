"""
Tool 7: Illustrative asset allocation rules (deterministic).

Step 1: start from the questionnaire risk category.
Step 2: apply the time horizon guardrail
          horizon < 3 years   -> Conservative
          horizon 3 to 5 years -> at most Moderate
          horizon > 5 years   -> unchanged
Step 3: Emergency Fund goals use a 100% debt / liquid allocation.
Step 4: weighted expected return = sum(weight x assumed asset return).

All percentages and returns are ILLUSTRATIVE ASSUMPTIONS from config.py.
"""

import config
from schemas.models import AllocationResult, GoalType, RiskCategory
from tools._validation import ensure_horizon_years, ensure_non_negative

ORDER = [RiskCategory.CONSERVATIVE, RiskCategory.MODERATE, RiskCategory.AGGRESSIVE]


def effective_category(category: RiskCategory, horizon_years: int) -> tuple[RiskCategory, bool, str | None]:
    """Apply the time horizon guardrail. Returns (category, applied?, reason)."""
    years = ensure_horizon_years(horizon_years)
    if years < config.GUARDRAIL_CONSERVATIVE_BELOW_YEARS and category != RiskCategory.CONSERVATIVE:
        return (
            RiskCategory.CONSERVATIVE,
            True,
            f"Goal horizon is {years} year(s), under {config.GUARDRAIL_CONSERVATIVE_BELOW_YEARS}. "
            "Money needed soon has little time to recover from a market fall, so the "
            "Conservative allocation is used regardless of the risk score.",
        )
    if years <= config.GUARDRAIL_MODERATE_UP_TO_YEARS and ORDER.index(category) > ORDER.index(RiskCategory.MODERATE):
        return (
            RiskCategory.MODERATE,
            True,
            f"Goal horizon is {years} years (up to {config.GUARDRAIL_MODERATE_UP_TO_YEARS}), "
            "so equity exposure is capped at the Moderate level.",
        )
    return category, False, None


def weighted_return(weights: dict[str, float], asset_returns: dict[str, float]) -> float:
    return sum(weights[a] * asset_returns[a] for a in ("equity", "debt", "gold")) / 100


def compute_allocation(
    risk_category: RiskCategory | str,
    horizon_years: int,
    goal_type: GoalType | str,
    monthly_expenses: float,
    monthly_debt: float,
    emergency_fund: float,
    asset_returns: dict[str, float] | None = None,
) -> AllocationResult:
    category = RiskCategory(risk_category)
    goal = GoalType(goal_type)
    returns = asset_returns or config.ASSUMED_ANNUAL_RETURNS_PCT

    eff, applied, reason = effective_category(category, horizon_years)
    if goal == GoalType.EMERGENCY_FUND:
        weights = config.EMERGENCY_GOAL_ALLOCATION
        eff, applied = RiskCategory.CONSERVATIVE, True
        reason = "Emergency fund money must stay stable and easy to withdraw, so a 100% debt/liquid allocation is used."
    else:
        weights = config.ALLOCATION_TABLES[eff.value]

    outflow = ensure_non_negative(monthly_expenses, "Monthly expenses") + ensure_non_negative(monthly_debt, "Monthly debt")
    reserve_target = config.EMERGENCY_RESERVE_MONTHS * outflow
    reserve_gap = max(reserve_target - ensure_non_negative(emergency_fund, "Emergency fund"), 0.0)

    return AllocationResult(
        risk_category=category,
        effective_category=eff,
        guardrail_applied=applied,
        guardrail_reason=reason,
        equity_pct=weights["equity"],
        debt_pct=weights["debt"],
        gold_pct=weights["gold"],
        weighted_return_pct=round(weighted_return(weights, returns), 4),
        emergency_reserve_gap=reserve_gap,
    )
