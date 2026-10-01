"""
AGENT 1: Profile & Goal Agent.

Deterministic part : cash flow (surplus), validation warnings, missing info.
LLM part           : reads the user's free text goal and extracts a
                     structured ParsedGoal; the result is cross checked
                     against the form, and any mismatch is reported.

The form values are always the ones used for calculations. The LLM output is
used to catch mistakes and to set goal priority. This agent never recommends
investments.
"""

from agents.llm import LLMCallError, LLMClient
from prompts.agent_prompts import PROFILE_GOAL_SYSTEM, PROFILE_GOAL_USER
from schemas.agent_outputs import ProfileAgentOutput
from schemas.models import GoalInput, ParsedGoal, UserProfile
from tools.financial_health import financial_health
from utils.formatting import format_inr
from utils.logging import SessionLog, Timer
from utils.validation import business_warnings

AGENT = "profile_agent"
AMOUNT_TOLERANCE = 0.10  # 10% difference between text and form is flagged


def parse_goal_text(text: str, llm: LLMClient, user_age: int | None = None) -> ParsedGoal:
    """LLM extraction of a goal from free text. Raises LLMCallError on failure."""
    user = PROFILE_GOAL_USER.format(age=user_age if user_age is not None else "not given", text=text)
    return llm.structured(ParsedGoal, PROFILE_GOAL_SYSTEM, user)


def compare_goal(parsed: ParsedGoal, goal: GoalInput) -> list[str]:
    """Differences between what the user wrote and what they entered in the form."""
    issues = []
    if parsed.goal_type != goal.goal_type:
        issues.append(
            f"Your description sounds like a '{parsed.goal_type.value}' goal, "
            f"but the form says '{goal.goal_type.value}'."
        )
    if parsed.goal_amount is not None:
        diff = abs(parsed.goal_amount - goal.goal_amount) / goal.goal_amount
        if diff > AMOUNT_TOLERANCE:
            issues.append(
                f"Your description mentions about {format_inr(parsed.goal_amount)}, "
                f"but the form says {format_inr(goal.goal_amount)}."
            )
    if parsed.horizon_years is not None and parsed.horizon_years != goal.horizon_years:
        issues.append(
            f"Your description suggests {parsed.horizon_years} years, "
            f"but the form says {goal.horizon_years} years."
        )
    if parsed.amount_is_present_value is not None and parsed.amount_is_present_value != goal.amount_is_present_value:
        said = "today's value" if parsed.amount_is_present_value else "future value"
        issues.append(f"Your description suggests the amount is in {said}; please check the form setting.")
    return issues


def run_profile_agent(
    profile: UserProfile,
    goal: GoalInput,
    llm: LLMClient | None,
    log: SessionLog | None = None,
) -> ProfileAgentOutput:
    health = financial_health(
        profile.monthly_income, profile.monthly_expenses, profile.monthly_debt,
        profile.monthly_investment, profile.emergency_fund,
    )
    warnings = business_warnings(profile, goal)

    missing: list[str] = []
    if profile.monthly_income == 0:
        missing.append("Monthly income is 0. Please confirm this is correct.")
    if not goal.description:
        missing.append("No goal description was given, so goal priority was not assessed.")

    out = ProfileAgentOutput(
        profile=profile,
        goal=goal,
        monthly_surplus=health.monthly_surplus,
        free_investable_surplus=health.free_investable_surplus,
        missing_information=missing,
        warnings=warnings,
    )

    if not goal.description:
        _log(log, "skipped goal parsing (no description)", "ok")
        return out
    if llm is None:
        out.llm_error = "LLM not configured."
        _log(log, "goal parsing skipped: no LLM", "fallback", out.llm_error)
        return out

    with Timer() as t:
        try:
            parsed = parse_goal_text(goal.description, llm, profile.age)
            out.parsed_goal = parsed
            out.goal_priority = parsed.priority
            out.discrepancies = compare_goal(parsed, goal)
            out.llm_used = True
        except LLMCallError as e:
            out.llm_error = str(e)
    _log(
        log,
        f"parsed goal: {out.parsed_goal.goal_type.value if out.parsed_goal else 'n/a'}, "
        f"{len(out.discrepancies)} discrepancies",
        "ok" if out.llm_used else "fallback",
        out.llm_error,
        t.ms,
    )
    return out


def _log(log, summary, status, error=None, ms=None):
    if log is not None:
        log.record(AGENT, "profile_and_goal", inputs={"profile": 1, "goal": 1},
                   output_summary=summary, status=status, error=error, duration_ms=ms)
