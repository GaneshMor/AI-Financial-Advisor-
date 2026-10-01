"""
Tool 3: Inflation calculator.

    Future cost = Present cost * (1 + inflation rate)^years

Inflation compounds ANNUALLY here, which is the standard way goal costs
are projected.
"""

from tools._validation import ensure_inflation_pct, ensure_non_negative


def inflation_adjusted_value(present_value: float, annual_inflation_pct: float, years: float) -> float:
    """Cost of a goal after `years` years of inflation."""
    pv = ensure_non_negative(present_value, "Present value")
    rate = ensure_inflation_pct(annual_inflation_pct)
    t = ensure_non_negative(years, "Years")
    return pv * (1 + rate / 100) ** t
