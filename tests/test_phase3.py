"""
Phase 3 tests: schemas, validation, risk scoring and the two datasets.
Run:  pytest -v
"""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import config
from schemas.models import (
    AllocationResult, GoalInput, GoalType, ParsedGoal, RiskCategory, SafetyCategory, ToolName, UserProfile,
)
from schemas.risk_questionnaire import QUESTIONS, parse_answers, score_answers
from tools import GoalPlanInputs, plan_goal
from utils.validation import split_user_row, validate_inputs

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


USERS = load("sample_users.csv")
CASES = load("test_cases.csv")

GOOD_PROFILE = dict(
    age=30, monthly_income=80000, monthly_expenses=40000, monthly_debt=10000, current_savings=100000,
    current_investments=200000, monthly_investment=10000, dependents=0, emergency_fund=300000,
)
GOOD_GOAL = dict(goal_type="House", goal_amount=2000000, horizon_years=5, current_goal_savings=100000)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
def test_valid_profile_and_goal():
    r = validate_inputs(GOOD_PROFILE, GOOD_GOAL)
    assert r.ok and r.warnings == []
    assert r.goal.amount_is_present_value is True  # default


@pytest.mark.parametrize("field, bad", [
    ("monthly_income", -1), ("age", 12), ("age", 101), ("dependents", -1), ("emergency_fund", -500),
])
def test_invalid_profile_values(field, bad):
    r = validate_inputs({**GOOD_PROFILE, field: bad}, GOOD_GOAL)
    assert not r.ok
    assert any(field.split("_")[0].lower() in e.lower() for e in r.errors)


@pytest.mark.parametrize("field, bad", [
    ("goal_amount", 0), ("horizon_years", 0), ("horizon_years", 51), ("goal_type", "Holiday"),
])
def test_invalid_goal_values(field, bad):
    assert not validate_inputs(GOOD_PROFILE, {**GOOD_GOAL, field: bad}).ok


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        UserProfile(**GOOD_PROFILE, salary=5)


def test_allocation_must_sum_to_100():
    base = dict(risk_category="Moderate", effective_category="Moderate", weighted_return_pct=10.1)
    AllocationResult(**base, equity_pct=60, debt_pct=30, gold_pct=10)
    with pytest.raises(ValidationError):
        AllocationResult(**base, equity_pct=60, debt_pct=30, gold_pct=20)


def test_parsed_goal_allows_missing_values():
    g = ParsedGoal(goal_type="Wealth Creation", missing_fields=["goal_amount", "horizon_years"])
    assert g.goal_amount is None and g.horizon_years is None


# ---------------------------------------------------------------------------
# Business warnings
# ---------------------------------------------------------------------------
def test_warning_negative_cash_flow():
    r = validate_inputs({**GOOD_PROFILE, "monthly_expenses": 75000}, GOOD_GOAL)
    assert r.ok and any("negative" in w.lower() for w in r.warnings)


def test_warning_goal_savings_exceed_assets():
    r = validate_inputs(GOOD_PROFILE, {**GOOD_GOAL, "current_goal_savings": 900000})
    assert any("goal savings" in w.lower() for w in r.warnings)


def test_warning_low_emergency_with_dependents():
    r = validate_inputs({**GOOD_PROFILE, "emergency_fund": 50000, "dependents": 2}, GOOD_GOAL)
    assert any("dependents" in w.lower() for w in r.warnings)


# ---------------------------------------------------------------------------
# Risk questionnaire
# ---------------------------------------------------------------------------
def test_questionnaire_shape():
    assert len(QUESTIONS) == 7
    assert all(len(q["options"]) == 3 for q in QUESTIONS)


@pytest.mark.parametrize("answers, score, category", [
    ([1] * 7, 7, "Conservative"),
    ([2, 2, 2, 2, 1, 1, 1], 11, "Conservative"),
    ([2, 2, 2, 2, 2, 1, 1], 12, "Moderate"),
    ([3, 3, 2, 2, 2, 2, 2], 16, "Moderate"),
    ([3, 3, 3, 2, 2, 2, 2], 17, "Aggressive"),
    ([3] * 7, 21, "Aggressive"),
])
def test_risk_bands(answers, score, category):
    r = score_answers(answers)
    assert r.score == score and r.category.value == category


def test_risk_is_deterministic():
    results = {score_answers([2, 3, 1, 2, 3, 2, 1]).model_dump_json() for _ in range(10)}
    assert len(results) == 1


@pytest.mark.parametrize("bad", [[1] * 6, [1] * 8, [0, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1], [True] * 7])
def test_risk_rejects_bad_answers(bad):
    with pytest.raises(ValueError):
        score_answers(bad)


# ---------------------------------------------------------------------------
# sample_users.csv
# ---------------------------------------------------------------------------
def test_at_least_20_users():
    assert len(USERS) >= 20
    assert len({u["user_id"] for u in USERS}) == len(USERS)


@pytest.mark.parametrize("row", USERS, ids=[u["user_id"] for u in USERS])
def test_every_user_is_valid(row):
    profile, goal = split_user_row(row)
    r = validate_inputs(profile, goal)
    assert r.ok, r.errors
    # stored risk label matches the scoring rules
    risk = score_answers(parse_answers(row["risk_answers"]))
    assert risk.category.value == row["risk_profile"]
    assert risk.score == int(row["risk_score"])


def test_all_goal_types_and_risk_levels_covered():
    assert {u["goal_type"] for u in USERS} == {g.value for g in GoalType}
    assert {u["risk_profile"] for u in USERS} == {c.value for c in RiskCategory}


def test_edge_case_users_raise_expected_warnings():
    by_id = {u["user_id"]: u for u in USERS}
    expectations = {"U09": "negative", "U10": "emergency fund covers only", "U21": "income is 0"}
    for uid, phrase in expectations.items():
        r = validate_inputs(*split_user_row(by_id[uid]))
        assert any(phrase in w.lower() for w in r.warnings), (uid, r.warnings)


# ---------------------------------------------------------------------------
# test_cases.csv
# ---------------------------------------------------------------------------
def test_case_counts():
    assert len(CASES) >= 20
    cats = {c["category"] for c in CASES}
    assert cats == {"calculation", "goal_parsing", "risk_classification",
                    "tool_selection", "rag_retrieval", "safety", "routing"}


def test_labels_use_valid_names():
    tool_names = {t.value for t in ToolName}
    safety_names = {s.value for s in SafetyCategory}
    goal_names = {g.value for g in GoalType}
    for c in CASES:
        exp = json.loads(c["expected_json"])
        if c["category"] == "tool_selection":
            assert exp["tool"] in tool_names
        elif c["category"] == "rag_retrieval":
            assert exp["expected_source"] in config.KNOWLEDGE_BASE_FILES
        elif c["category"] == "safety":
            assert set(exp["flags"]) <= safety_names
        elif c["category"] == "goal_parsing":
            assert exp["goal_type"] in goal_names


def test_risk_cases_match_scoring():
    for c in (c for c in CASES if c["category"] == "risk_classification"):
        exp = json.loads(c["expected_json"])
        r = score_answers(parse_answers(json.loads(c["input_json"])["answers"]))
        assert (r.score, r.category.value) == (exp["score"], exp["category"])


def test_calculation_cases_match_tools():
    """Preview of the Phase 9 metric: tools vs independent ground truth, tolerance Rs 1."""
    calc = [c for c in CASES if c["category"] == "calculation"]
    assert len(calc) >= 10
    for c in calc:
        exp = json.loads(c["expected_json"])
        res = plan_goal(GoalPlanInputs(**json.loads(c["input_json"])))
        assert res.target_future_value == pytest.approx(exp["target_future_value"], abs=1), c["case_id"]
        assert res.fv_current_goal_savings == pytest.approx(exp["fv_current_goal_savings"], abs=1), c["case_id"]
        assert res.required_monthly_sip == pytest.approx(exp["required_monthly_sip"], abs=1), c["case_id"]
        assert res.projected_value_current_plan == pytest.approx(exp["projected_value_current_plan"], abs=1), c["case_id"]
        assert res.gap.status == exp["status"], c["case_id"]
