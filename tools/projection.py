"""
Year by year projection of a goal account (used for charts).

Same maths as the other tools: monthly SIP at the start of each month,
monthly compounding at annual % / 12. Starts from current goal savings.
"""

from tools._validation import ensure_months, ensure_non_negative
from tools.future_value import lumpsum_future_value
from tools.sip_calculator import sip_future_value


def yearly_projection(current_savings: float, monthly_sip: float, annual_return_pct: float,
                      years: int) -> list[dict]:
    """[{year, value, invested}] for year 0..years."""
    savings = ensure_non_negative(current_savings, "Current savings")
    sip = ensure_non_negative(monthly_sip, "Monthly SIP")
    n_years = ensure_months(years, "Years")
    rows = []
    for y in range(n_years + 1):
        m = y * 12
        value = lumpsum_future_value(savings, annual_return_pct, m) + sip_future_value(sip, annual_return_pct, m)
        rows.append({"year": y, "value": value, "invested": savings + sip * m})
    return rows
