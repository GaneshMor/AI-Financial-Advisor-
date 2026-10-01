"""
Pydantic data models shared by the whole application.

Why Pydantic?
  * Every agent input and output has a fixed, validated shape.
  * LLM outputs are parsed into these models, so a malformed LLM reply
    fails validation instead of silently flowing into the plan.

Money is in Indian rupees. Rates are in percent (12 means 12%).
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config

NonNegMoney = Field(ge=0, description="Amount in INR, 0 or more")


class _Strict(BaseModel):
    """Base model: reject unknown fields so typos in agent JSON are caught."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class GoalType(str, Enum):
    RETIREMENT = "Retirement"
    HOUSE = "House"
    EDUCATION = "Education"
    MARRIAGE = "Marriage"
    VEHICLE = "Vehicle"
    EMERGENCY_FUND = "Emergency Fund"
    WEALTH_CREATION = "Wealth Creation"
    OTHER = "Other"


class GoalPriority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RiskCategory(str, Enum):
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


class ToolName(str, Enum):
    """Names the Q&A agent uses when calling tools (Phase 4 wrappers use these)."""

    REQUIRED_SIP = "calculate_required_sip"
    FUTURE_VALUE = "calculate_future_value"
    INFLATION = "calculate_inflation"
    GOAL_GAP = "calculate_goal_gap"
    FINANCIAL_HEALTH = "calculate_financial_health"
    SCENARIO = "run_what_if_scenario"


class SafetyCategory(str, Enum):
    GUARANTEED_RETURN = "guaranteed_return"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    RISK_MISMATCH = "risk_mismatch"
    HALLUCINATION = "hallucination"
    MISSING_ASSUMPTION = "missing_assumption"
    ESCALATION = "escalation"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
class UserProfile(_Strict):
    age: int = Field(ge=18, le=100)
    monthly_income: float = NonNegMoney
    monthly_expenses: float = Field(ge=0, description="Living costs only. Excludes EMI and investments.")
    monthly_debt: float = Field(ge=0, description="Total monthly EMIs")
    current_savings: float = Field(ge=0, description="Bank and cash savings, excluding the emergency fund")
    current_investments: float = Field(ge=0, description="Market investments: MF, stocks, PPF, etc.")
    monthly_investment: float = Field(ge=0, description="Current monthly SIP / investment")
    dependents: int = Field(ge=0, le=20)
    emergency_fund: float = Field(ge=0, description="Money kept aside only for emergencies")


class GoalInput(_Strict):
    goal_type: GoalType
    goal_amount: float = Field(gt=0)
    amount_is_present_value: bool = Field(
        default=True, description="True = amount in today's rupees, will be inflation adjusted"
    )
    current_goal_savings: float = Field(default=0, ge=0)
    horizon_years: int = Field(ge=1, le=config.MAX_HORIZON_YEARS)
    description: str | None = Field(default=None, max_length=500, description="Goal in the user's own words")


class Assumptions(_Strict):
    """Every assumption shown to the user. All are illustrative."""

    expected_return_pct: float = Field(ge=config.MIN_RETURN_PCT, le=config.MAX_RETURN_PCT)
    inflation_pct: float = Field(ge=config.MIN_INFLATION_PCT, le=config.MAX_INFLATION_PCT)
    asset_returns_pct: dict[str, float] = Field(default_factory=lambda: dict(config.ASSUMED_ANNUAL_RETURNS_PCT))
    contribution_frequency: str = config.CONTRIBUTION_FREQUENCY
    label: str = config.ASSUMPTION_LABEL


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------
class ParsedGoal(_Strict):
    """What the Profile Agent's LLM extracts from the free text goal description."""

    goal_type: GoalType
    goal_amount: float | None = Field(default=None, gt=0)
    horizon_years: int | None = Field(default=None, ge=1, le=config.MAX_HORIZON_YEARS)
    amount_is_present_value: bool | None = None
    priority: GoalPriority = GoalPriority.MEDIUM
    missing_fields: list[str] = Field(default_factory=list)


class RiskResult(_Strict):
    answers: list[int] = Field(min_length=7, max_length=7)
    score: int = Field(ge=7, le=21)
    category: RiskCategory
    points_by_question: dict[str, int]
    reasoning: str | None = Field(default=None, description="LLM explanation (Phase 4)")
    key_risk_factors: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Academic prototype risk classification. Not a regulated suitability assessment."
    )


class AllocationResult(_Strict):
    """Illustrative allocation of the MONTHLY investment. Must sum to 100%."""

    risk_category: RiskCategory               # from the questionnaire
    effective_category: RiskCategory          # after the time horizon guardrail
    guardrail_applied: bool = False
    guardrail_reason: str | None = None
    equity_pct: float = Field(ge=0, le=100)
    debt_pct: float = Field(ge=0, le=100)
    gold_pct: float = Field(ge=0, le=100)
    weighted_return_pct: float = Field(ge=0)
    emergency_reserve_gap: float = Field(default=0, ge=0, description="INR still needed for a 6 month reserve")
    explanation: str | None = None
    label: str = "Illustrative allocation for education. Not a recommendation of any product."

    @model_validator(mode="after")
    def _sums_to_100(self):
        total = self.equity_pct + self.debt_pct + self.gold_pct
        if abs(total - 100) > 0.01:
            raise ValueError(f"Allocation must sum to 100%, got {total}%.")
        return self


class RetrievedChunk(_Strict):
    text: str
    source_file: str
    title: str
    source_org: str | None = None
    source_url: str | None = None
    score: float


class SafetyFlag(_Strict):
    category: SafetyCategory
    severity: Severity
    text: str = Field(description="The problem sentence or a description of what is missing")
    suggestion: str | None = None


class SafetyReport(_Strict):
    flags: list[SafetyFlag] = Field(default_factory=list)
    checked_by_llm: bool = False

    @property
    def passed(self) -> bool:
        return not any(f.severity == Severity.HIGH for f in self.flags)
