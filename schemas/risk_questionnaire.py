"""
Risk profiling questionnaire and transparent scoring.

7 questions x 1 to 3 points = score from 7 to 21.
    7  to 11 -> Conservative
    12 to 16 -> Moderate
    17 to 21 -> Aggressive

Scoring is deterministic Python: the same answers ALWAYS give the same score.
The LLM (Phase 4) only explains the result; it never changes the score.

This is an academic prototype classification, not a regulated suitability
assessment.
"""

from schemas.models import RiskCategory, RiskResult

QUESTIONS: list[dict] = [
    {
        "id": "q1_horizon",
        "text": "How long until you need most of the money you are investing?",
        "options": ["Less than 3 years", "3 to 7 years", "More than 7 years"],
    },
    {
        "id": "q2_market_decline",
        "text": "If your investments fell 20% in one year, what would you most likely do?",
        "options": ["Sell to avoid further loss", "Wait and do nothing", "Invest more at lower prices"],
    },
    {
        "id": "q3_income_stability",
        "text": "How stable is your income?",
        "options": ["Irregular or uncertain", "Mostly stable", "Very stable with growth prospects"],
    },
    {
        "id": "q4_obligations",
        "text": "What share of your monthly income goes to EMIs and fixed obligations?",
        "options": ["More than 40%", "20% to 40%", "Less than 20%"],
    },
    {
        "id": "q5_emergency_fund",
        "text": "How many months of expenses does your emergency fund cover?",
        "options": ["Less than 3 months", "3 to 6 months", "More than 6 months"],
    },
    {
        "id": "q6_experience",
        "text": "What is your investment experience?",
        "options": ["None, only savings account or FDs", "Some mutual funds", "Experienced with equity"],
    },
    {
        "id": "q7_preference",
        "text": "Which statement fits you best?",
        "options": [
            "Protect my capital, even if returns are low",
            "Balance growth and safety",
            "Maximise growth, I accept large ups and downs",
        ],
    },
]

# Option index 0, 1, 2 -> points 1, 2, 3
POINTS = [1, 2, 3]

BANDS: list[tuple[int, int, RiskCategory]] = [
    (7, 11, RiskCategory.CONSERVATIVE),
    (12, 16, RiskCategory.MODERATE),
    (17, 21, RiskCategory.AGGRESSIVE),
]


def category_for_score(score: int) -> RiskCategory:
    for low, high, category in BANDS:
        if low <= score <= high:
            return category
    raise ValueError(f"Score must be between 7 and 21, got {score}.")


def score_answers(answers: list[int]) -> RiskResult:
    """
    answers: list of 7 points, each 1, 2 or 3 (in question order).
    Returns the score, category and a per question breakdown.
    """
    if len(answers) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} answers, got {len(answers)}.")
    for i, a in enumerate(answers, start=1):
        if isinstance(a, bool) or a not in POINTS:
            raise ValueError(f"Answer {i} must be 1, 2 or 3 (got {a!r}).")

    score = sum(answers)
    return RiskResult(
        answers=list(answers),
        score=score,
        category=category_for_score(score),
        points_by_question={q["id"]: a for q, a in zip(QUESTIONS, answers)},
    )


def parse_answers(text: str) -> list[int]:
    """Parse '3,2,2,1,3,2,3' (the CSV format) into a list of ints."""
    return [int(part.strip()) for part in str(text).split(",")]


# ---------------------------------------------------------------------------
# Suggested answers derived from the user's own numbers (Q1, Q4, Q5).
# The UI pre-selects these; the user can still change them.
# ---------------------------------------------------------------------------
def derive_q1(horizon_years: int) -> int:
    """Horizon: under 3 years -> 1, 3 to 7 -> 2, more than 7 -> 3."""
    return 1 if horizon_years < 3 else 2 if horizon_years <= 7 else 3


def derive_q4(income: float, emi: float) -> int:
    """EMI share of income: over 40% -> 1, 20 to 40% -> 2, under 20% -> 3. No income -> 1."""
    if income == 0:
        return 1
    share = emi / income * 100
    return 1 if share > 40 else 2 if share >= 20 else 3


def derive_q5(emergency: float, expenses: float, emi: float) -> int:
    """Emergency months: under 3 -> 1, 3 to 6 -> 2, over 6 -> 3."""
    outflow = expenses + emi
    months = emergency / outflow if outflow else 99
    return 1 if months < 3 else 2 if months <= 6 else 3


def suggested_answers(profile: dict | None, goal: dict | None) -> dict[int, int]:
    """{question_index: suggested points} for the questions answerable from the numbers."""
    out: dict[int, int] = {}
    if goal and goal.get("horizon_years"):
        out[0] = derive_q1(int(goal["horizon_years"]))
    if profile:
        out[3] = derive_q4(float(profile["monthly_income"]), float(profile["monthly_debt"]))
        out[4] = derive_q5(float(profile["emergency_fund"]), float(profile["monthly_expenses"]),
                           float(profile["monthly_debt"]))
    return out
