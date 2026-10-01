"""
Phase 2 tests for the calculation tools.

Ground truth is computed INDEPENDENTLY of the tool code:
  * SIP values by simulating the account month by month (no formula)
  * fixed values worked out by hand and written into the test

Run:  pytest -v
"""

import math

import pytest

from tools import (
    GoalPlanInputs,
    financial_health,
    goal_gap,
    inflation_adjusted_value,
    lumpsum_future_value,
    plan_goal,
    required_monthly_sip,
    run_scenario,
    sip_future_value,
    standard_scenarios,
    total_future_value,
)
from utils.formatting import format_inr, format_lakh_crore


# ---------------------------------------------------------------------------
# Independent reference: simulate a SIP month by month
# ---------------------------------------------------------------------------
def simulate_sip(monthly: float, annual_pct: float, months: int) -> float:
    """Deposit at the start of each month, then apply one month of growth."""
    r = annual_pct / 100 / 12
    balance = 0.0
    for _ in range(months):
        balance = (balance + monthly) * (1 + r)
    return balance


# ---------------------------------------------------------------------------
# Tool 1: SIP
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "monthly, annual, months",
    [(5000, 12, 60), (10000, 10, 120), (2500, 8, 36), (15000, 7, 240), (1000, 0.5, 12)],
)
def test_sip_future_value_matches_simulation(monthly, annual, months):
    assert sip_future_value(monthly, annual, months) == pytest.approx(
        simulate_sip(monthly, annual, months), rel=1e-9
    )


def test_required_sip_known_value():
    # Rs 10,00,000 in 5 years at 12%: factor = 82.48637, SIP = Rs 12,123.22
    assert required_monthly_sip(1_000_000, 12, 60) == pytest.approx(12123.22, abs=0.01)


def test_required_sip_round_trip():
    sip = required_monthly_sip(2_500_000, 11, 96)
    assert simulate_sip(sip, 11, 96) == pytest.approx(2_500_000, rel=1e-9)


def test_zero_return_cases():
    assert sip_future_value(5000, 0, 24) == 120_000
    assert required_monthly_sip(120_000, 0, 24) == 5000


def test_zero_target_and_zero_months():
    assert required_monthly_sip(0, 12, 60) == 0
    assert sip_future_value(5000, 12, 0) == 0
    with pytest.raises(ValueError):
        required_monthly_sip(100_000, 12, 0)


# ---------------------------------------------------------------------------
# Tool 2: Future value
# ---------------------------------------------------------------------------
def test_lumpsum_monthly_compounding():
    # 1,00,000 at 12% for 12 months, monthly compounding: 1.01^12 = 1.126825...
    assert lumpsum_future_value(100_000, 12, 12) == pytest.approx(112_682.50, abs=0.01)


def test_total_future_value_breakdown():
    fv = total_future_value(200_000, 5000, 10, 120)
    assert fv.fv_lumpsum == pytest.approx(200_000 * (1 + 0.10 / 12) ** 120)
    assert fv.fv_sip == pytest.approx(simulate_sip(5000, 10, 120))
    assert fv.total_future_value == pytest.approx(fv.fv_lumpsum + fv.fv_sip)
    assert fv.total_invested == 200_000 + 5000 * 120
    assert fv.estimated_gain == pytest.approx(fv.total_future_value - fv.total_invested)


# ---------------------------------------------------------------------------
# Tool 3: Inflation
# ---------------------------------------------------------------------------
def test_inflation_known_value():
    # 1.06^10 = 1.790847697 -> Rs 17,90,847.70
    assert inflation_adjusted_value(1_000_000, 6, 10) == pytest.approx(1_790_847.70, abs=0.01)


def test_inflation_zero_rate_and_zero_years():
    assert inflation_adjusted_value(500_000, 0, 10) == 500_000
    assert inflation_adjusted_value(500_000, 6, 0) == 500_000


# ---------------------------------------------------------------------------
# Tool 4: Goal gap and full plan
# ---------------------------------------------------------------------------
def test_goal_gap_statuses():
    shortfall = goal_gap(12_000, 5_000, 10_000)
    assert shortfall.status == "shortfall"
    assert shortfall.monthly_gap == 7_000
    assert shortfall.gap_affordable is True
    assert shortfall.uncovered_gap == 0

    unaffordable = goal_gap(12_000, 5_000, 3_000)
    assert unaffordable.gap_affordable is False
    assert unaffordable.uncovered_gap == 4_000

    ahead = goal_gap(5_000, 8_000, 0)
    assert ahead.status == "ahead" and ahead.monthly_gap == -3_000

    on_track = goal_gap(5_000.4, 5_000, 0)
    assert on_track.status == "on_track"


def test_goal_gap_negative_surplus():
    result = goal_gap(10_000, 4_000, -2_000)
    assert result.uncovered_gap == 6_000
    assert result.gap_affordable is False


def test_plan_goal_present_value_goal():
    inputs = GoalPlanInputs(
        goal_amount=1_000_000, amount_is_present_value=True, horizon_years=10,
        annual_inflation_pct=6, annual_return_pct=12,
        current_goal_savings=100_000, current_monthly_sip=3_000, free_investable_surplus=10_000,
    )
    res = plan_goal(inputs)
    target = 1_000_000 * 1.06 ** 10
    grown = 100_000 * (1.01) ** 120
    assert res.months == 120
    assert res.target_future_value == pytest.approx(target)
    assert res.fv_current_goal_savings == pytest.approx(grown)
    assert simulate_sip(res.required_monthly_sip, 12, 120) == pytest.approx(target - grown, rel=1e-9)
    assert res.projected_value_current_plan == pytest.approx(grown + simulate_sip(3_000, 12, 120))


def test_plan_goal_future_value_goal_is_not_inflated():
    inputs = GoalPlanInputs(1_000_000, False, 5, 6, 12)
    assert plan_goal(inputs).target_future_value == 1_000_000


def test_plan_goal_consistency_required_sip_hits_target():
    """If the user already invests exactly the required SIP, projection equals target."""
    base = GoalPlanInputs(2_000_000, True, 8, 6, 10, current_goal_savings=50_000)
    required = plan_goal(base).required_monthly_sip
    res = plan_goal(GoalPlanInputs(2_000_000, True, 8, 6, 10, 50_000, required, 0))
    assert res.projected_surplus_or_shortfall == pytest.approx(0, abs=1e-4)
    assert res.gap.status == "on_track"


def test_plan_goal_already_funded():
    res = plan_goal(GoalPlanInputs(500_000, False, 5, 6, 10, current_goal_savings=600_000))
    assert res.net_target < 0
    assert res.required_monthly_sip == 0
    assert res.projected_surplus_or_shortfall > 0


# ---------------------------------------------------------------------------
# Tool 5: Financial health
# ---------------------------------------------------------------------------
def test_financial_health_ratios():
    h = financial_health(100_000, 50_000, 20_000, 10_000, 280_000)
    assert h.monthly_surplus == 30_000
    assert h.free_investable_surplus == 20_000
    assert h.savings_rate.value == pytest.approx(30.0)
    assert h.expense_ratio.value == pytest.approx(50.0)
    assert h.debt_to_income.value == pytest.approx(20.0)
    assert h.emergency_fund_coverage.value == pytest.approx(4.0)
    assert h.savings_rate.status == "Healthy"
    assert h.debt_to_income.status == "Comfortable"
    assert h.emergency_fund_coverage.status == "Partial"


def test_financial_health_zero_income():
    h = financial_health(0, 20_000, 0, 0, 50_000)
    assert h.savings_rate.value is None
    assert h.monthly_surplus == -20_000
    assert any("negative" in w.lower() for w in h.warnings)


def test_financial_health_zero_outflow():
    h = financial_health(50_000, 0, 0, 0, 10_000)
    assert h.emergency_fund_coverage.value is None


# ---------------------------------------------------------------------------
# Tool 6: Scenarios
# ---------------------------------------------------------------------------
def test_standard_scenarios_order():
    base = GoalPlanInputs(1_000_000, True, 10, 6, 10, current_monthly_sip=5_000)
    rows = standard_scenarios(base, 2)
    assert [r["scenario"] for r in rows] == ["Conservative", "Base", "Optimistic"]
    assert [r["annual_return_pct"] for r in rows] == [8, 10, 12]
    # Higher assumed return -> lower required SIP
    assert rows[0]["required_monthly_sip"] > rows[1]["required_monthly_sip"] > rows[2]["required_monthly_sip"]
    assert all("Not a forecast" in r["label"] for r in rows)


def test_scenario_rejects_unknown_override():
    base = GoalPlanInputs(1_000_000, True, 10, 6, 10)
    with pytest.raises(ValueError):
        run_scenario(base, "bad", goal_amount=5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda: sip_future_value(-1, 12, 12),
        lambda: sip_future_value(1000, -5, 12),
        lambda: sip_future_value(1000, 60, 12),
        lambda: sip_future_value(1000, 12, 12.5),
        lambda: inflation_adjusted_value(1000, 40, 5),
        lambda: financial_health(-1, 0, 0, 0, 0),
        lambda: plan_goal(GoalPlanInputs(1_000_000, True, 0, 6, 10)),
        lambda: plan_goal(GoalPlanInputs(1_000_000, True, 5.5, 6, 10)),
        lambda: required_monthly_sip(math.nan, 12, 12),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("bad", ["1000", None, True])
def test_non_numbers_raise_type_error(bad):
    with pytest.raises(TypeError):
        sip_future_value(bad, 12, 12)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_format_inr():
    assert format_inr(1234567) == "₹12,34,567"
    assert format_inr(12123.1) == "₹12,123"
    assert format_inr(999) == "₹999"
    assert format_inr(-12500000) == "-₹1,25,00,000"
    assert format_inr(-0.4) == "₹0"
    assert format_inr(1790847.697, 2) == "₹17,90,847.70"


def test_format_lakh_crore():
    assert format_lakh_crore(12_500_000) == "₹1.25 Cr"
    assert format_lakh_crore(450_000) == "₹4.50 L"
    assert format_lakh_crore(9_500) == "₹9,500"
