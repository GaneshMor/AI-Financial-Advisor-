"""
Tool 1: SIP calculator.

Formula (SIP paid at the START of each month, i.e. an annuity due):

    FV = P * [((1 + r)^n - 1) / r] * (1 + r)

    P = monthly SIP amount
    r = monthly rate = annual return % / 100 / 12
    n = number of months

When r = 0 the formula divides by zero, so we use FV = P * n instead.
"""

from tools._validation import ensure_months, ensure_non_negative, ensure_return_pct


def monthly_rate(annual_return_pct: float) -> float:
    """Convert an annual return in percent (e.g. 12) to a monthly decimal rate (0.01)."""
    return ensure_return_pct(annual_return_pct) / 100 / 12


def _annuity_due_factor(r: float, n: int) -> float:
    """Future value of Rs 1 invested at the start of every month for n months."""
    if n == 0:
        return 0.0
    if r == 0:
        return float(n)
    return (((1 + r) ** n - 1) / r) * (1 + r)


def sip_future_value(monthly_sip: float, annual_return_pct: float, months: int) -> float:
    """Future value of a monthly SIP after `months` months."""
    p = ensure_non_negative(monthly_sip, "Monthly SIP")
    n = ensure_months(months)
    r = monthly_rate(annual_return_pct)
    return p * _annuity_due_factor(r, n)


def required_monthly_sip(target_amount: float, annual_return_pct: float, months: int) -> float:
    """
    Monthly SIP needed to reach `target_amount` in `months` months.

    Returns 0 if the target is 0. Raises ValueError if a positive target
    must be reached in 0 months (impossible).
    """
    target = ensure_non_negative(target_amount, "Target amount")
    n = ensure_months(months)
    r = monthly_rate(annual_return_pct)

    if target == 0:
        return 0.0
    if n == 0:
        raise ValueError("A positive target cannot be reached in 0 months.")
    return target / _annuity_due_factor(r, n)
