"""
Tool 2: Future value calculator.

Lump sum growth uses the SAME monthly rate as the SIP calculator
(annual % / 12, compounded monthly) so that both parts of a plan are
consistent. Note: at 12% this means an effective annual rate of about
12.68%, because (1 + 0.12/12)^12 - 1 = 0.1268.
"""

from dataclasses import asdict, dataclass

from tools._validation import ensure_months, ensure_non_negative
from tools.sip_calculator import monthly_rate, sip_future_value


def lumpsum_future_value(amount: float, annual_return_pct: float, months: int) -> float:
    """Future value of a one time amount after `months` months."""
    principal = ensure_non_negative(amount, "Lump sum amount")
    n = ensure_months(months)
    r = monthly_rate(annual_return_pct)
    return principal * (1 + r) ** n


@dataclass(frozen=True)
class FutureValueBreakdown:
    lumpsum_invested: float
    monthly_sip: float
    months: int
    annual_return_pct: float
    fv_lumpsum: float
    fv_sip: float
    total_future_value: float
    total_invested: float
    estimated_gain: float

    def to_dict(self) -> dict:
        return asdict(self)


def total_future_value(
    existing_amount: float,
    monthly_sip: float,
    annual_return_pct: float,
    months: int,
) -> FutureValueBreakdown:
    """Future value of existing investments plus a monthly SIP."""
    lump = ensure_non_negative(existing_amount, "Existing investments")
    sip = ensure_non_negative(monthly_sip, "Monthly SIP")
    n = ensure_months(months)

    fv_lump = lumpsum_future_value(lump, annual_return_pct, n)
    fv_sip = sip_future_value(sip, annual_return_pct, n)
    invested = lump + sip * n
    total = fv_lump + fv_sip

    return FutureValueBreakdown(
        lumpsum_invested=lump,
        monthly_sip=sip,
        months=n,
        annual_return_pct=float(annual_return_pct),
        fv_lumpsum=fv_lump,
        fv_sip=fv_sip,
        total_future_value=total,
        total_invested=invested,
        estimated_gain=total - invested,
    )
