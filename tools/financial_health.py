"""
Tool 5: Financial health calculator.

Definitions (fixed in Phase 1):
  Monthly expenses       = living costs only (NOT EMI, NOT investments)
  Monthly surplus        = Income - Expenses - EMI
  Free investable surplus= Surplus - current monthly investment
  Savings rate %         = Surplus / Income * 100
  Expense ratio %        = Expenses / Income * 100
  Debt to income %       = EMI / Income * 100
  Emergency fund months  = Emergency fund / (Expenses + EMI)

These are PROTOTYPE INDICATORS, not universal financial rules.
If income is 0, the ratios cannot be calculated and are returned as None.
"""

from dataclasses import asdict, dataclass

import config
from tools._validation import ensure_non_negative


@dataclass(frozen=True)
class HealthIndicator:
    name: str
    value: float | None
    unit: str
    status: str   # e.g. "Healthy", "Moderate", "Low", or "Not available"
    note: str


@dataclass(frozen=True)
class FinancialHealthResult:
    monthly_income: float
    monthly_expenses: float
    monthly_debt: float
    monthly_investment: float
    monthly_surplus: float
    free_investable_surplus: float
    savings_rate: HealthIndicator
    expense_ratio: HealthIndicator
    debt_to_income: HealthIndicator
    emergency_fund_coverage: HealthIndicator
    warnings: list[str]
    disclaimer: str

    def to_dict(self) -> dict:
        return asdict(self)


def _grade_higher_is_better(value: float, good: float, fair: float, labels: tuple[str, str, str]) -> str:
    if value >= good:
        return labels[0]
    if value >= fair:
        return labels[1]
    return labels[2]


def _grade_lower_is_better(value: float, good: float, fair: float, labels: tuple[str, str, str]) -> str:
    if value <= good:
        return labels[0]
    if value <= fair:
        return labels[1]
    return labels[2]


def financial_health(
    monthly_income: float,
    monthly_expenses: float,
    monthly_debt: float,
    monthly_investment: float,
    emergency_fund: float,
) -> FinancialHealthResult:
    income = ensure_non_negative(monthly_income, "Monthly income")
    expenses = ensure_non_negative(monthly_expenses, "Monthly expenses")
    debt = ensure_non_negative(monthly_debt, "Monthly debt / EMI")
    investment = ensure_non_negative(monthly_investment, "Monthly investment")
    emergency = ensure_non_negative(emergency_fund, "Emergency fund")

    t = config.HEALTH_THRESHOLDS
    surplus = income - expenses - debt
    free_surplus = surplus - investment
    warnings: list[str] = []

    if income == 0:
        na = "Not available"
        msg = "Income is 0, so income based ratios cannot be calculated."
        savings_rate = HealthIndicator("Savings rate", None, "%", na, msg)
        expense_ratio = HealthIndicator("Expense ratio", None, "%", na, msg)
        dti = HealthIndicator("Debt to income", None, "%", na, msg)
        warnings.append(msg)
    else:
        sr = surplus / income * 100
        er = expenses / income * 100
        dt = debt / income * 100
        savings_rate = HealthIndicator(
            "Savings rate", sr, "%",
            _grade_higher_is_better(sr, t["savings_rate_pct"]["good"], t["savings_rate_pct"]["fair"],
                                    ("Healthy", "Moderate", "Low")),
            "Share of income left after living expenses and EMI.",
        )
        expense_ratio = HealthIndicator(
            "Expense ratio", er, "%", "Info",
            "Share of income spent on living expenses (excludes EMI).",
        )
        dti = HealthIndicator(
            "Debt to income", dt, "%",
            _grade_lower_is_better(dt, t["debt_to_income_pct"]["good"], t["debt_to_income_pct"]["fair"],
                                   ("Comfortable", "Elevated", "High")),
            "Share of income going to EMIs.",
        )

    essential_outflow = expenses + debt
    if essential_outflow == 0:
        efc = HealthIndicator(
            "Emergency fund coverage", None, "months", "Not available",
            "Expenses and EMI are both 0, so coverage cannot be calculated.",
        )
    else:
        months = emergency / essential_outflow
        efc = HealthIndicator(
            "Emergency fund coverage", months, "months",
            _grade_higher_is_better(months, t["emergency_fund_months"]["good"],
                                    t["emergency_fund_months"]["fair"],
                                    ("Adequate", "Partial", "Low")),
            "Months of expenses plus EMI the emergency fund can cover.",
        )

    if surplus < 0:
        warnings.append("Expenses plus EMI exceed income. Monthly cash flow is negative.")
    elif free_surplus < 0:
        warnings.append("Current monthly investment is higher than the monthly surplus.")

    return FinancialHealthResult(
        monthly_income=income,
        monthly_expenses=expenses,
        monthly_debt=debt,
        monthly_investment=investment,
        monthly_surplus=surplus,
        free_investable_surplus=free_surplus,
        savings_rate=savings_rate,
        expense_ratio=expense_ratio,
        debt_to_income=dti,
        emergency_fund_coverage=efc,
        warnings=warnings,
        disclaimer=config.HEALTH_DISCLAIMER,
    )
