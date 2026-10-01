"""
Input validation with friendly messages.

Two levels:
  * errors   : input cannot be used (negative money, age 12, missing field).
               Comes from Pydantic.
  * warnings : input is valid but looks risky or inconsistent
               (negative cash flow, goal savings larger than total assets).
               The plan still runs; warnings are shown to the user and passed
               to the agents.
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from schemas.models import GoalInput, GoalType, UserProfile

FIELD_LABELS = {
    "age": "Age",
    "monthly_income": "Monthly income",
    "monthly_expenses": "Monthly expenses",
    "monthly_debt": "Monthly debt / EMI",
    "current_savings": "Current savings",
    "current_investments": "Current investments",
    "monthly_investment": "Monthly investment",
    "dependents": "Number of dependents",
    "emergency_fund": "Emergency fund",
    "goal_type": "Goal type",
    "goal_amount": "Goal amount",
    "amount_is_present_value": "Goal amount basis",
    "current_goal_savings": "Current goal savings",
    "horizon_years": "Time horizon",
    "description": "Goal description",
}


@dataclass
class ValidationReport:
    profile: UserProfile | None = None
    goal: GoalInput | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _friendly(err: ValidationError) -> list[str]:
    messages = []
    for e in err.errors():
        name = str(e["loc"][0]) if e["loc"] else "Input"
        label = FIELD_LABELS.get(name, name)
        messages.append(f"{label}: {e['msg']}")
    return messages


def business_warnings(p: UserProfile, g: GoalInput | None) -> list[str]:
    """Checks that do not block the plan but must be shown."""
    w: list[str] = []
    outflow = p.monthly_expenses + p.monthly_debt
    surplus = p.monthly_income - outflow

    if p.monthly_income == 0:
        w.append("Monthly income is 0. Ratios based on income cannot be calculated.")
    if surplus < 0:
        w.append("Expenses plus EMI are higher than income. Monthly cash flow is negative.")
    elif p.monthly_investment > surplus:
        w.append("Current monthly investment is higher than the monthly surplus.")

    if outflow > 0:
        months = p.emergency_fund / outflow
        if months < 3:
            extra = " You have dependents, so this matters more." if p.dependents > 0 else ""
            w.append(f"Emergency fund covers only {months:.1f} months of expenses plus EMI.{extra}")

    if p.monthly_income > 0 and p.monthly_debt / p.monthly_income > 0.4:
        w.append("EMIs are more than 40% of income.")

    if g is not None:
        if p.age + g.horizon_years > 100:
            w.append("Age plus goal horizon is more than 100 years. Please check the horizon.")
        if g.current_goal_savings > p.current_savings + p.current_investments:
            w.append(
                "Current goal savings are larger than total savings plus investments. "
                "Goal savings should be part of those amounts."
            )
        if g.goal_type == GoalType.EMERGENCY_FUND and g.horizon_years > 3:
            w.append("An emergency fund goal usually has a short horizon. Please check.")
        if g.goal_type == GoalType.RETIREMENT and p.age + g.horizon_years < 45:
            w.append("Retirement age implied by the horizon is below 45. Please check.")
    return w


def validate_inputs(profile_data: dict, goal_data: dict | None = None) -> ValidationReport:
    """Validate raw dicts (from the UI form or a CSV row)."""
    report = ValidationReport()
    try:
        report.profile = UserProfile(**profile_data)
    except ValidationError as e:
        report.errors.extend(_friendly(e))

    if goal_data is not None:
        try:
            report.goal = GoalInput(**goal_data)
        except ValidationError as e:
            report.errors.extend(_friendly(e))

    if report.profile is not None:
        report.warnings = business_warnings(report.profile, report.goal)
    return report


PROFILE_COLUMNS = [
    "age", "monthly_income", "monthly_expenses", "monthly_debt", "current_savings",
    "current_investments", "monthly_investment", "dependents", "emergency_fund",
]


def split_user_row(row: dict) -> tuple[dict, dict]:
    """Turn one sample_users.csv row into (profile_data, goal_data)."""
    profile = {k: row[k] for k in PROFILE_COLUMNS}
    goal = {
        "goal_type": row["goal_type"],
        "goal_amount": row["goal_amount"],
        "amount_is_present_value": str(row.get("amount_is_present_value", True)).strip().lower()
        in {"true", "1", "yes"},
        "current_goal_savings": row["current_goal_savings"],
        "horizon_years": row["goal_horizon_years"],
        "description": row.get("goal_description") or None,
    }
    return profile, goal
