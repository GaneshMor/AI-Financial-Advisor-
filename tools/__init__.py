"""Deterministic financial calculation tools. No LLM is used in this package."""

from tools.allocation import compute_allocation, effective_category
from tools.financial_health import FinancialHealthResult, financial_health
from tools.future_value import FutureValueBreakdown, lumpsum_future_value, total_future_value
from tools.goal_gap import GoalGapResult, GoalPlanInputs, GoalPlanResult, goal_gap, plan_goal
from tools.inflation import inflation_adjusted_value
from tools.scenario import SCENARIO_LABEL, run_scenario, standard_scenarios
from tools.sip_calculator import monthly_rate, required_monthly_sip, sip_future_value

__all__ = [
    "compute_allocation", "effective_category",
    "FinancialHealthResult", "financial_health",
    "FutureValueBreakdown", "lumpsum_future_value", "total_future_value",
    "GoalGapResult", "GoalPlanInputs", "GoalPlanResult", "goal_gap", "plan_goal",
    "inflation_adjusted_value",
    "SCENARIO_LABEL", "run_scenario", "standard_scenarios",
    "monthly_rate", "required_monthly_sip", "sip_future_value",
]
