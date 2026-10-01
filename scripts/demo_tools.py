"""
Phase 2 demo: runs every calculation tool on one sample user.
Run from the project root:   python -m scripts.demo_tools
"""

import config
from tools import GoalPlanInputs, financial_health, plan_goal, standard_scenarios
from utils.formatting import format_inr

# Sample user (synthetic)
income, expenses, emi, sip, emergency = 90_000, 45_000, 12_000, 8_000, 150_000

health = financial_health(income, expenses, emi, sip, emergency)
print("=== FINANCIAL HEALTH (prototype indicators) ===")
print("Monthly surplus        :", format_inr(health.monthly_surplus))
print("Free investable surplus:", format_inr(health.free_investable_surplus))
for ind in (health.savings_rate, health.expense_ratio, health.debt_to_income, health.emergency_fund_coverage):
    value = "n/a" if ind.value is None else f"{ind.value:.1f} {ind.unit}"
    print(f"{ind.name:<24}: {value:<12} [{ind.status}]")
for w in health.warnings:
    print("WARNING:", w)

inputs = GoalPlanInputs(
    goal_amount=2_500_000,             # house down payment in today's rupees
    amount_is_present_value=True,
    horizon_years=7,
    annual_inflation_pct=config.DEFAULT_INFLATION_PCT,
    annual_return_pct=10.1,            # Moderate weighted return (illustrative)
    current_goal_savings=300_000,
    current_monthly_sip=sip,
    free_investable_surplus=health.free_investable_surplus,
)
plan = plan_goal(inputs)
print("\n=== GOAL PLAN ===")
print("Target (inflation adjusted):", format_inr(plan.target_future_value))
print("Current savings grow to    :", format_inr(plan.fv_current_goal_savings))
print("Required monthly SIP       :", format_inr(plan.required_monthly_sip))
print("Current monthly SIP        :", format_inr(plan.gap.current_monthly_sip))
print("Monthly gap                :", format_inr(plan.gap.monthly_gap), f"({plan.gap.status})")
print("Gap affordable from surplus:", plan.gap.gap_affordable)
print("Assumptions:", plan.assumptions)

print("\n=== WHAT IF? (hypothetical, not forecasts) ===")
for row in standard_scenarios(inputs):
    print(f"{row['scenario']:<13} return {row['annual_return_pct']:>5.1f}%  "
          f"required SIP {format_inr(row['required_monthly_sip']):>10}  "
          f"projected {format_inr(row['projected_value_current_plan']):>13}  {row['status']}")
