"""
Tool 4: Goal gap calculator, plus `plan_goal`, which chains tools 1 to 4.

Definitions (fixed in Phase 1):
  Target (future value) = goal amount, inflated if entered in today's value
  Net target            = Target - future value of current goal savings
  Required SIP          = SIP needed to reach the net target
  Monthly gap           = Required SIP - current monthly SIP
                          (> 0 shortfall, < 0 ahead of plan)

Prototype assumption: the user's current monthly investment is treated as
fully dedicated to this one goal.
"""

from dataclasses import asdict, dataclass

import config
from tools._validation import (
    ensure_horizon_years,
    ensure_inflation_pct,
    ensure_non_negative,
    ensure_number,
    ensure_return_pct,
)
from tools.future_value import lumpsum_future_value
from tools.inflation import inflation_adjusted_value
from tools.sip_calculator import required_monthly_sip, sip_future_value


# ---------------------------------------------------------------------------
# Tool 4: goal gap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoalGapResult:
    required_monthly_sip: float
    current_monthly_sip: float
    monthly_gap: float               # required - current
    status: str                      # "shortfall" | "on_track" | "ahead"
    free_investable_surplus: float   # can be negative
    gap_affordable: bool             # can the free surplus cover the gap?
    uncovered_gap: float             # part of the gap the surplus cannot cover

    def to_dict(self) -> dict:
        return asdict(self)


def goal_gap(
    required_sip: float,
    current_sip: float,
    free_investable_surplus: float,
) -> GoalGapResult:
    """Compare required SIP with current SIP and spare monthly cash."""
    required = ensure_non_negative(required_sip, "Required SIP")
    current = ensure_non_negative(current_sip, "Current SIP")
    surplus = ensure_number(free_investable_surplus, "Free investable surplus")

    gap = required - current
    if abs(gap) <= config.ON_TRACK_TOLERANCE_INR:
        status = "on_track"
    elif gap > 0:
        status = "shortfall"
    else:
        status = "ahead"

    usable_surplus = max(surplus, 0.0)
    positive_gap = max(gap, 0.0) if status == "shortfall" else 0.0
    uncovered = max(positive_gap - usable_surplus, 0.0)

    return GoalGapResult(
        required_monthly_sip=required,
        current_monthly_sip=current,
        monthly_gap=gap,
        status=status,
        free_investable_surplus=surplus,
        gap_affordable=uncovered == 0.0,
        uncovered_gap=uncovered,
    )


# ---------------------------------------------------------------------------
# Full goal plan (chains inflation -> future value -> SIP -> gap)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GoalPlanInputs:
    goal_amount: float
    amount_is_present_value: bool      # True = in today's rupees, will be inflated
    horizon_years: int
    annual_inflation_pct: float
    annual_return_pct: float
    current_goal_savings: float = 0.0
    current_monthly_sip: float = 0.0
    free_investable_surplus: float = 0.0


@dataclass(frozen=True)
class GoalPlanResult:
    months: int
    target_future_value: float           # what the goal will cost at the deadline
    fv_current_goal_savings: float       # what today's goal savings grow to
    net_target: float                    # target minus grown savings (can be <= 0)
    required_monthly_sip: float
    projected_value_current_plan: float  # grown savings + current SIP
    projected_surplus_or_shortfall: float  # projected - target (negative = shortfall)
    gap: GoalGapResult
    assumptions: dict

    def to_dict(self) -> dict:
        return asdict(self)


def plan_goal(inputs: GoalPlanInputs) -> GoalPlanResult:
    """Run the full deterministic goal calculation."""
    goal_amount = ensure_non_negative(inputs.goal_amount, "Goal amount")
    years = ensure_horizon_years(inputs.horizon_years)
    inflation = ensure_inflation_pct(inputs.annual_inflation_pct)
    ret = ensure_return_pct(inputs.annual_return_pct)
    savings = ensure_non_negative(inputs.current_goal_savings, "Current goal savings")
    current_sip = ensure_non_negative(inputs.current_monthly_sip, "Current monthly SIP")
    months = years * 12

    if inputs.amount_is_present_value:
        target = inflation_adjusted_value(goal_amount, inflation, years)
    else:
        target = goal_amount

    fv_savings = lumpsum_future_value(savings, ret, months)
    net_target = target - fv_savings
    required = required_monthly_sip(max(net_target, 0.0), ret, months)

    projected = fv_savings + sip_future_value(current_sip, ret, months)
    gap = goal_gap(required, current_sip, inputs.free_investable_surplus)

    return GoalPlanResult(
        months=months,
        target_future_value=target,
        fv_current_goal_savings=fv_savings,
        net_target=net_target,
        required_monthly_sip=required,
        projected_value_current_plan=projected,
        projected_surplus_or_shortfall=projected - target,
        gap=gap,
        assumptions={
            "expected_annual_return_pct": ret,
            "annual_inflation_pct": inflation if inputs.amount_is_present_value else None,
            "goal_amount_is_present_value": inputs.amount_is_present_value,
            "horizon_years": years,
            "contribution_frequency": config.CONTRIBUTION_FREQUENCY,
            "current_sip_dedicated_to_goal": True,
            "label": config.ASSUMPTION_LABEL,
        },
    )
