"""
AGENT 2: Risk Profiling Agent.

Deterministic part : questionnaire score (7 to 21) and category. Same answers
                     always give the same result.
LLM part           : explains the result and lists key risk factors.
                     The LLM cannot change the score or the category.

Fallback (no LLM)  : key risk factors are listed from the answers that scored
                     1 point, clearly marked as rule based.
"""

import json

from agents.llm import LLMCallError, LLMClient
from prompts.agent_prompts import RISK_SYSTEM, RISK_USER
from schemas.agent_outputs import RiskAgentOutput, RiskExplanationLLM
from schemas.models import GoalInput, UserProfile
from schemas.risk_questionnaire import BANDS, QUESTIONS, score_answers
from utils.logging import SessionLog, Timer

AGENT = "risk_agent"

LOW_SCORE_FACTORS = {
    "q1_horizon": "Short investment horizon (under 3 years)",
    "q2_market_decline": "Likely to sell during a market fall",
    "q3_income_stability": "Irregular or uncertain income",
    "q4_obligations": "EMIs and fixed obligations above 40% of income",
    "q5_emergency_fund": "Emergency fund below 3 months",
    "q6_experience": "Limited investment experience",
    "q7_preference": "Strong preference for protecting capital",
}


def rule_based_factors(points_by_question: dict[str, int]) -> list[str]:
    return [LOW_SCORE_FACTORS[q] for q, p in points_by_question.items() if p == 1]


def risk_facts(result, profile: UserProfile | None, goal: GoalInput | None) -> dict:
    answers = [
        {"question": q["text"], "answer": q["options"][a - 1], "points": a}
        for q, a in zip(QUESTIONS, result.answers)
    ]
    facts = {
        "score": result.score,
        "category": result.category.value,
        "bands": [f"{lo} to {hi}: {cat.value}" for lo, hi, cat in BANDS],
        "answers": answers,
    }
    if profile is not None:
        facts["age"] = profile.age
        facts["dependents"] = profile.dependents
    if goal is not None:
        facts["goal_type"] = goal.goal_type.value
        facts["goal_horizon_years"] = goal.horizon_years
    return facts


def run_risk_agent(
    answers: list[int],
    llm: LLMClient | None,
    profile: UserProfile | None = None,
    goal: GoalInput | None = None,
    log: SessionLog | None = None,
) -> RiskAgentOutput:
    result = score_answers(answers)  # raises ValueError on bad answers
    out = RiskAgentOutput(risk=result)

    if llm is None:
        out.llm_error = "LLM not configured."
    else:
        with Timer() as t:
            try:
                facts = json.dumps(risk_facts(result, profile, goal), indent=2)
                exp = llm.structured(RiskExplanationLLM, RISK_SYSTEM, RISK_USER.format(facts=facts))
                out.risk = result.model_copy(
                    update={"reasoning": exp.reasoning, "key_risk_factors": exp.key_risk_factors}
                )
                out.factors_source = "llm"
                out.llm_used = True
            except LLMCallError as e:
                out.llm_error = str(e)

    if not out.llm_used:
        out.risk = result.model_copy(update={"key_risk_factors": rule_based_factors(result.points_by_question)})

    if log is not None:
        log.record(AGENT, "score_and_explain", tool="score_answers", inputs={"answers": 1},
                   output_summary=f"score {result.score}, {result.category.value}",
                   status="ok" if out.llm_used else "fallback", error=out.llm_error,
                   duration_ms=None if llm is None else t.ms)
    return out
