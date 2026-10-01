"""
LangChain wrappers so the Q&A agent can CALL the calculators (tool calling).

The LLM only chooses the tool and fills the arguments. The maths is always
done by the Python functions in this package. Tool names come from
schemas.models.ToolName so evaluation labels and code stay in sync.

Errors (bad arguments) are returned as {"error": "..."} so the LLM can tell
the user what went wrong instead of crashing.
"""

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

import config
from schemas.models import ToolName
from tools.financial_health import financial_health
from tools.future_value import total_future_value
from tools.goal_gap import GoalPlanInputs, plan_goal
from tools.inflation import inflation_adjusted_value
from tools.scenario import SCENARIO_LABEL, run_scenario
from tools.sip_calculator import required_monthly_sip


def _r(x):
    return round(x, 2) if isinstance(x, float) else x


def _safe(fn):
    def wrapper(**kwargs):
        try:
            return fn(**kwargs)
        except (ValueError, TypeError) as e:
            return {"error": str(e)}
    return wrapper


# ---------------------------------------------------------------------------
# Argument schemas (descriptions help the LLM pick the right tool)
# ---------------------------------------------------------------------------
class RequiredSipArgs(BaseModel):
    target_amount: float = Field(description="Amount needed at the end, in rupees")
    annual_return_pct: float = Field(description="Assumed annual return in percent, e.g. 10.1")
    horizon_years: int = Field(description="Years until the money is needed")


class FutureValueArgs(BaseModel):
    annual_return_pct: float = Field(description="Assumed annual return in percent")
    horizon_years: int = Field(description="Number of years invested")
    monthly_sip: float = Field(default=0, description="Monthly SIP in rupees")
    existing_amount: float = Field(default=0, description="Lump sum already invested, in rupees")


class InflationArgs(BaseModel):
    present_value: float = Field(description="Cost today, in rupees")
    annual_inflation_pct: float = Field(description="Assumed annual inflation in percent")
    years: int = Field(description="Number of years into the future")


class GoalPlanArgs(BaseModel):
    goal_amount: float = Field(description="Goal amount in rupees")
    amount_is_present_value: bool = Field(description="True if the goal amount is in today's rupees")
    horizon_years: int = Field(description="Years until the goal")
    annual_inflation_pct: float = Field(description="Assumed annual inflation in percent")
    annual_return_pct: float = Field(description="Assumed annual return in percent")
    current_goal_savings: float = Field(default=0, description="Savings already set aside for this goal")
    current_monthly_sip: float = Field(default=0, description="Current monthly investment toward the goal")
    free_investable_surplus: float = Field(default=0, description="Spare monthly cash after expenses, EMI and SIP")


class HealthArgs(BaseModel):
    monthly_income: float
    monthly_expenses: float = Field(description="Living expenses, excluding EMI")
    monthly_debt: float = Field(description="Total monthly EMI")
    monthly_investment: float
    emergency_fund: float


class ScenarioArgs(GoalPlanArgs):
    new_annual_return_pct: float | None = Field(default=None, description="Changed return to test")
    new_annual_inflation_pct: float | None = Field(default=None, description="Changed inflation to test")
    new_monthly_sip: float | None = Field(default=None, description="Changed monthly SIP to test")
    new_horizon_years: int | None = Field(default=None, description="Changed horizon to test")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------
@_safe
def _required_sip(target_amount, annual_return_pct, horizon_years):
    sip = required_monthly_sip(target_amount, annual_return_pct, int(horizon_years) * 12)
    return {"required_monthly_sip": _r(sip), "assumption": config.ASSUMPTION_LABEL}


@_safe
def _future_value(annual_return_pct, horizon_years, monthly_sip=0, existing_amount=0):
    fv = total_future_value(existing_amount, monthly_sip, annual_return_pct, int(horizon_years) * 12)
    return {k: _r(v) for k, v in fv.to_dict().items()} | {"assumption": config.ASSUMPTION_LABEL}


@_safe
def _inflation(present_value, annual_inflation_pct, years):
    return {"future_cost": _r(inflation_adjusted_value(present_value, annual_inflation_pct, years)),
            "assumption": config.ASSUMPTION_LABEL}


def _plan_summary(res) -> dict:
    return {
        "target_future_value": _r(res.target_future_value),
        "fv_current_goal_savings": _r(res.fv_current_goal_savings),
        "required_monthly_sip": _r(res.required_monthly_sip),
        "current_monthly_sip": _r(res.gap.current_monthly_sip),
        "monthly_gap": _r(res.gap.monthly_gap),
        "status": res.gap.status,
        "gap_affordable_from_surplus": res.gap.gap_affordable,
        "uncovered_gap": _r(res.gap.uncovered_gap),
        "projected_value_current_plan": _r(res.projected_value_current_plan),
        "assumptions": res.assumptions,
    }


@_safe
def _goal_gap(**kwargs):
    return _plan_summary(plan_goal(GoalPlanInputs(**kwargs)))


@_safe
def _health(**kwargs):
    h = financial_health(**kwargs)
    out = {"monthly_surplus": _r(h.monthly_surplus), "free_investable_surplus": _r(h.free_investable_surplus)}
    for ind in (h.savings_rate, h.expense_ratio, h.debt_to_income, h.emergency_fund_coverage):
        out[ind.name] = {"value": _r(ind.value), "unit": ind.unit, "status": ind.status}
    out["warnings"] = h.warnings
    out["disclaimer"] = h.disclaimer
    return out


@_safe
def _scenario(new_annual_return_pct=None, new_annual_inflation_pct=None, new_monthly_sip=None,
              new_horizon_years=None, **base_kwargs):
    base = GoalPlanInputs(**base_kwargs)
    overrides = {
        k: v for k, v in {
            "annual_return_pct": new_annual_return_pct,
            "annual_inflation_pct": new_annual_inflation_pct,
            "current_monthly_sip": new_monthly_sip,
            "horizon_years": new_horizon_years,
        }.items() if v is not None
    }
    base_row = run_scenario(base, "Current assumptions")
    what_if = run_scenario(base, "What if", **overrides)
    return {"base": {k: _r(v) for k, v in base_row.items()},
            "what_if": {k: _r(v) for k, v in what_if.items()},
            "label": SCENARIO_LABEL}


def get_lc_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            _required_sip, name=ToolName.REQUIRED_SIP.value, args_schema=RequiredSipArgs,
            description="Monthly SIP needed to reach a target amount in a given number of years.",
        ),
        StructuredTool.from_function(
            _future_value, name=ToolName.FUTURE_VALUE.value, args_schema=FutureValueArgs,
            description="Future value of a monthly SIP and/or a lump sum after some years.",
        ),
        StructuredTool.from_function(
            _inflation, name=ToolName.INFLATION.value, args_schema=InflationArgs,
            description="Future cost of something that costs a given amount today, after inflation.",
        ),
        StructuredTool.from_function(
            _goal_gap, name=ToolName.GOAL_GAP.value, args_schema=GoalPlanArgs,
            description="Check the user's goal: required SIP, current SIP, monthly gap, and whether "
                        "the goal is on track. Use for 'can I reach my goal' or 'how much more should I invest'.",
        ),
        StructuredTool.from_function(
            _health, name=ToolName.FINANCIAL_HEALTH.value, args_schema=HealthArgs,
            description="Savings rate, expense ratio, debt to income ratio and emergency fund coverage.",
        ),
        StructuredTool.from_function(
            _scenario, name=ToolName.SCENARIO.value, args_schema=ScenarioArgs,
            description="What if analysis: re-run the goal plan with a changed return, inflation, SIP "
                        "or horizon and compare with current assumptions.",
        ),
    ]


def tools_by_name() -> dict[str, StructuredTool]:
    return {t.name: t for t in get_lc_tools()}
