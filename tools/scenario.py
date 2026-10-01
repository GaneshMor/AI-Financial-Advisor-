"""
Tool 6: "What If?" scenario runner.

Re-runs `plan_goal` with changed assumptions. Results are HYPOTHETICAL
scenarios, not forecasts.
"""

from dataclasses import replace

from tools.goal_gap import GoalPlanInputs, plan_goal

SCENARIO_LABEL = "Hypothetical scenario based on assumed rates. Not a forecast."

ALLOWED_OVERRIDES = {
    "annual_return_pct",
    "annual_inflation_pct",
    "current_monthly_sip",
    "horizon_years",
}


def run_scenario(base: GoalPlanInputs, name: str, **overrides) -> dict:
    """Run one scenario. Only the fields in ALLOWED_OVERRIDES can be changed."""
    unknown = set(overrides) - ALLOWED_OVERRIDES
    if unknown:
        raise ValueError(f"Cannot override: {sorted(unknown)}. Allowed: {sorted(ALLOWED_OVERRIDES)}")

    inputs = replace(base, **overrides)
    result = plan_goal(inputs)
    return {
        "scenario": name,
        "annual_return_pct": inputs.annual_return_pct,
        "annual_inflation_pct": inputs.annual_inflation_pct,
        "current_monthly_sip": inputs.current_monthly_sip,
        "horizon_years": inputs.horizon_years,
        "target_future_value": result.target_future_value,
        "required_monthly_sip": result.required_monthly_sip,
        "projected_value_current_plan": result.projected_value_current_plan,
        "projected_surplus_or_shortfall": result.projected_surplus_or_shortfall,
        "status": result.gap.status,
        "label": SCENARIO_LABEL,
    }


def standard_scenarios(base: GoalPlanInputs, return_spread_pct: float = 2.0) -> list[dict]:
    """Conservative / Base / Optimistic: expected return minus, equal to, plus the spread."""
    if return_spread_pct < 0:
        raise ValueError("Return spread cannot be negative.")
    low = max(base.annual_return_pct - return_spread_pct, 0.0)
    high = base.annual_return_pct + return_spread_pct
    return [
        run_scenario(base, "Conservative", annual_return_pct=low),
        run_scenario(base, "Base"),
        run_scenario(base, "Optimistic", annual_return_pct=high),
    ]
