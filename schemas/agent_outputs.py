"""
Output schemas for each agent.

Two kinds:
  * ...LLM models : what the LLM must return (kept simple so function calling
                    works reliably). Parsed and validated by Pydantic.
  * ...Output     : what the agent returns to the orchestrator. Contains the
                    deterministic results, the LLM part (or None if the LLM
                    failed) and the error message, so failures are visible.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.models import (
    AllocationResult, Assumptions, GoalInput, GoalPriority, ParsedGoal, RetrievedChunk, RiskResult, UserProfile,
)


class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# LLM output models
# ---------------------------------------------------------------------------
class RiskExplanationLLM(_LLMModel):
    reasoning: str = Field(description="3 to 5 sentences explaining why the score gives this category")
    key_risk_factors: list[str] = Field(description="2 to 5 short factors from the user's answers and numbers")


class PortfolioExplanationLLM(_LLMModel):
    why_selected: str = Field(description="Why this illustrative allocation was selected")
    risk_effect: str = Field(description="How the risk category affects the allocation")
    horizon_effect: str = Field(description="How the time horizon affects the allocation")
    remaining_risks: list[str] = Field(description="Risks that remain even with this allocation")


class RagAnswerLLM(_LLMModel):
    answer: str = Field(description="Answer using ONLY the provided context")
    sources_used: list[str] = Field(description="File names from the context that support the answer")
    answerable: bool = Field(description="False if the context does not contain the answer")


class RouteLLM(_LLMModel):
    route: Literal["calculation", "knowledge", "out_of_scope"] = Field(
        description="calculation = needs numbers from a calculator; knowledge = concept or explanation; "
                    "out_of_scope = stock tips, specific products, tax filing, legal, or unrelated")
    reason: str = Field(description="One short sentence")


class SafetyIssueLLM(_LLMModel):
    category: Literal["unsupported_claim", "hallucination", "guaranteed_return", "risk_mismatch"]
    severity: Literal["medium", "high"]
    text: str = Field(description="The exact problem sentence")
    suggestion: str = Field(description="How to fix it")


class SafetyJudgeLLM(_LLMModel):
    issues: list[SafetyIssueLLM] = Field(default_factory=list)


class ClaimJudgementLLM(_LLMModel):
    claim: str = Field(description="One factual claim from the text, quoted or closely paraphrased")
    supported: bool = Field(description="True only if the FACTS or CONTEXT support it")


class GroundednessLLM(_LLMModel):
    claims: list[ClaimJudgementLLM] = Field(default_factory=list)


class RiskItemLLM(_LLMModel):
    name: str = Field(description="e.g. Market risk, Inflation risk, Liquidity risk")
    explanation: str


class AdvisorNarrativeLLM(_LLMModel):
    summary: str = Field(description="4 to 6 sentence plain language summary of the plan")
    allocation_explanation: str = Field(description="Why this illustrative allocation fits the facts")
    action_plan: list[str] = Field(description="3 to 7 concrete actions based only on the facts")
    risk_explanations: list[RiskItemLLM] = Field(description="Market, inflation, liquidity, interest rate (if relevant), goal shortfall")
    sources_cited: list[str] = Field(default_factory=list, description="Context file names actually used")


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------
class ToolCallRecord(BaseModel):
    tool: str
    inputs: dict
    output: dict | list | None = None
    success: bool = True
    error: str | None = None


class ProfileAgentOutput(BaseModel):
    profile: UserProfile
    goal: GoalInput
    monthly_surplus: float
    free_investable_surplus: float
    parsed_goal: ParsedGoal | None = None
    discrepancies: list[str] = Field(default_factory=list)
    goal_priority: GoalPriority | None = None
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_error: str | None = None


class RiskAgentOutput(BaseModel):
    risk: RiskResult
    factors_source: str = "rules"   # "llm" or "rules"
    llm_used: bool = False
    llm_error: str | None = None


class PortfolioAgentOutput(BaseModel):
    allocation: AllocationResult
    explanation: PortfolioExplanationLLM | None = None
    llm_used: bool = False
    llm_error: str | None = None


class CalculationAgentOutput(BaseModel):
    assumptions: Assumptions
    health: dict
    goal_plan: dict
    scenarios: list[dict]
    tool_calls: list[ToolCallRecord]


class QAToolAnswer(BaseModel):
    question: str
    answer: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_error: str | None = None


class RagAgentOutput(BaseModel):
    question: str
    answer: str | None = None
    answerable: bool = False
    sources_used: list[str] = Field(default_factory=list)
    invented_sources_removed: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    llm_error: str | None = None


class FinalPlan(BaseModel):
    snapshot: dict
    goal: dict
    risk: dict
    goal_calculation: dict
    allocation: dict
    scenarios: list[dict]
    narrative: AdvisorNarrativeLLM | None = None
    assumptions: dict
    warnings: list[str] = Field(default_factory=list)
    professional_advice_recommended: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    disclaimer: str
    llm_error: str | None = None


class QAResult(BaseModel):
    question: str
    route: str | None = None
    route_reason: str | None = None
    answer: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    invented_sources_removed: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
