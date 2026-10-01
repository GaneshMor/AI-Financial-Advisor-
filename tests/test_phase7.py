"""
Phase 7 tests: the Safety Agent.

1. The 8 labelled safety cases from data/test_cases.csv.
2. UNSEEN clean texts (must not be flagged) and UNSEEN bad texts (must be
   flagged), written separately so the rules are not only tuned to the labels.
3. Plan checks (structure, numbers, escalation) and the LLM judge.
4. Integration with both LangGraph graphs.
"""

import csv
import json
from pathlib import Path

import pytest

from agents.safety_agent import (
    check_numbers, check_plan, check_text, collect_numbers, extract_amounts, make_plan_safety_fn, make_qa_safety_fn,
)
from graph.plan_graph import MAX_REVISIONS, run_plan
from graph.qa_graph import ask
from rag.retriever import KeywordRetriever
from schemas.agent_outputs import AdvisorNarrativeLLM, RagAnswerLLM, RiskItemLLM, RouteLLM, SafetyIssueLLM, SafetyJudgeLLM
from tests.fakes import FakeLLM
from utils.formatting import format_inr

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "data" / "test_cases.csv").open(encoding="utf-8") as f:
    SAFETY_CASES = [c for c in csv.DictReader(f) if c["category"] == "safety"]

PROFILE = dict(age=30, monthly_income=85000, monthly_expenses=38000, monthly_debt=15000,
               current_savings=150000, current_investments=300000, monthly_investment=15000,
               dependents=0, emergency_fund=250000)
GOAL = dict(goal_type="House", goal_amount=2500000, horizon_years=6, current_goal_savings=400000)
ANSWERS = [2, 2, 3, 3, 2, 2, 2]


# ---------------------------------------------------------------------------
# 1. Labelled cases (rules only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", SAFETY_CASES, ids=[c["case_id"] for c in SAFETY_CASES])
def test_labelled_safety_cases(case):
    inp = json.loads(case["input_json"])
    expected = set(json.loads(case["expected_json"])["flags"])
    report = check_text(inp["text"], inp.get("context"))
    got = {f.category.value for f in report.flags}
    assert expected <= got
    if not expected:
        assert report.passed and not report.flags


# ---------------------------------------------------------------------------
# 2. Unseen texts
# ---------------------------------------------------------------------------
CLEAN = [
    "Returns are not guaranteed, and the value of equity funds can fall.",
    "Your Emergency Fund goal uses a 100% debt allocation because the money must stay liquid.",
    "A Systematic Investment Plan invests a fixed amount every month.",
    "Gold ETFs track the price of gold, but gold prices can fall.",
    "Government bonds are considered risk free from default, but their prices still change with interest rates.",
    "There is no guarantee that the goal will be reached.",
    "Consider increasing your SIP if your budget allows, and review the plan every year.",
    "Mutual fund investments are subject to market risks.",
    "Assuming an illustrative 12% annual return, Rs 10,000 a month for 10 years may grow to about Rs 23 lakh.",
    "SEBI recommends reviewing your investments regularly.",
    "Sovereign Gold Bonds are one option mentioned by NCFE.",
]
BAD = [
    ("You are guaranteed to reach Rs 50 lakh.", "guaranteed_return"),
    ("This SIP will earn 14% every year.", "guaranteed_return"),
    ("The HDFC Top 100 Fund is a good choice for you.", "hallucination"),
    ("Put your money in Axis Bluechip Fund.", "hallucination"),
    ("Nifty has historically returned 12% a year.", "hallucination"),
    ("RBI mandates that you hold an emergency fund.", "hallucination"),
    ("Equity always beats inflation over the long run.", "unsupported_claim"),
    ("Gold never falls in value.", "unsupported_claim"),
    ("Your money is 100% safe in this plan.", "unsupported_claim"),
    ("Hybrid funds are best (source: morningstar_report.pdf).", "hallucination"),
    ("Invest Rs 20,000 per month and you will have Rs 1.2 crore in 15 years.", "missing_assumption"),
]


@pytest.mark.parametrize("text", CLEAN)
def test_clean_text_not_flagged(text):
    assert check_text(text, valid_sources={"sip.md"}).flags == []


@pytest.mark.parametrize("text, category", BAD)
def test_bad_text_flagged(text, category):
    report = check_text(text, valid_sources={"sip.md"})
    assert category in {f.category.value for f in report.flags} and not report.passed


# ---------------------------------------------------------------------------
# Number checks
# ---------------------------------------------------------------------------
def test_amount_extraction_units():
    vals = sorted(v for v, _, _ in extract_amounts("₹37,59,076 or Rs. 25.5 lakh or 1.2 crore or INR 500"))
    assert vals == [500, 2550000, 3759076, 12000000]


def test_number_check_accepts_rounded_and_flags_invented():
    allowed = {3759076.0, 25748.0, 10.1, 6.0}
    assert check_numbers("Target is ₹37,59,076 and about 37.6 lakh; SIP ₹25,748 at 10.1% and 6% inflation.", allowed) == []
    flags = check_numbers("You will need ₹45,000 a month.", allowed)
    assert flags and "45,000" in flags[0].suggestion
    assert check_numbers("Returns of 18% are assumed.", allowed)


# ---------------------------------------------------------------------------
# 3. Plan checks
# ---------------------------------------------------------------------------
def _plan(**overrides):
    return run_plan(PROFILE, GOAL, ANSWERS, None, **overrides)


def _narrative_from(plan, summary, actions=None):
    gc = plan.goal_calculation
    return AdvisorNarrativeLLM(
        summary=summary,
        allocation_explanation="The illustrative Moderate allocation is 60% equity, 30% debt and 10% gold.",
        action_plan=actions or [f"Increase your SIP by about {format_inr(gc['monthly_gap'])} if your budget allows.",
                                "Review progress every year."],
        risk_explanations=[RiskItemLLM(name="Market risk", explanation="Equity values can fall.")],
    )


def test_structure_only_plan_passes():
    res = _plan()
    assert check_plan(res.plan).flags == []


def test_tampered_allocation_is_flagged():
    plan = _plan().plan
    bad = plan.model_copy(update={"allocation": dict(plan.allocation, equity_pct=90, debt_pct=5, gold_pct=5)})
    assert "risk_mismatch" in {f.category.value for f in check_plan(bad).flags}


def test_good_narrative_passes_number_check():
    plan = _plan().plan
    gc = plan.goal_calculation
    n = _narrative_from(plan, f"Assuming an illustrative {plan.assumptions['expected_annual_return_pct']}% return, "
                              f"you need about {format_inr(gc['required_monthly_sip'])} a month to reach "
                              f"{format_inr(gc['target_future_value'])}.")
    report = check_plan(plan.model_copy(update={"narrative": n}), use_llm=False)
    assert report.flags == [], report.flags


def test_invented_number_in_narrative_is_flagged():
    plan = _plan().plan
    n = _narrative_from(plan, "Assuming an illustrative return, you need about ₹41,500 a month.")
    cats = {f.category.value for f in check_plan(plan.model_copy(update={"narrative": n}), use_llm=False).flags}
    assert "hallucination" in cats


def test_escalation_required_when_rules_say_so():
    plan = _plan().plan.model_copy(update={"professional_advice_recommended": True,
                                           "escalation_reasons": ["EMIs are more than 40% of income."]})
    n = _narrative_from(plan, "Assuming an illustrative return, the plan is achievable.", ["Review yearly."])
    flags = check_plan(plan.model_copy(update={"narrative": n}), use_llm=False).flags
    assert "escalation" in {f.category.value for f in flags}


def test_llm_judge_adds_flags_and_failure_is_safe():
    issue = SafetyIssueLLM(category="unsupported_claim", severity="high",
                           text="Gold protects you in every crisis.", suggestion="Remove")
    r = check_text("Gold protects you in every crisis.", llm=FakeLLM({"SafetyJudgeLLM": SafetyJudgeLLM(issues=[issue])}))
    assert r.checked_by_llm and r.flags[0].suggestion.startswith("[LLM judge]")
    r2 = check_text("Gold protects you in every crisis.", llm=FakeLLM(fail=True))
    assert not r2.checked_by_llm and r2.flags == []  # rules alone did not catch this one


def test_collect_numbers_reads_nested_and_strings():
    nums = collect_numbers({"a": 1.5, "b": [2, {"c": "₹3,00,000 and 7%"}], "d": True})
    assert {1.5, 2.0, 300000.0, 7.0} <= nums and 1.0 not in nums


# ---------------------------------------------------------------------------
# 4. Graph integration
# ---------------------------------------------------------------------------
def test_plan_graph_rewrites_then_withholds_unsafe_narrative():
    bad = AdvisorNarrativeLLM(summary="This plan will definitely give you 15% returns.", allocation_explanation="x",
                              action_plan=["a", "b", "c"], risk_explanations=[])
    llm = FakeLLM({"AdvisorNarrativeLLM": bad})
    res = run_plan(PROFILE, GOAL, ANSWERS, llm, safety_fn=make_plan_safety_fn(llm, use_llm=False))
    assert res.ok and res.revision_count == MAX_REVISIONS
    assert res.safety_status == "failed_after_revisions"
    assert res.plan.narrative is None and "withheld" in res.plan.llm_error
    assert res.plan.goal_calculation["required_monthly_sip"] > 0  # numbers still delivered


def test_plan_graph_accepts_fixed_narrative_after_revision():
    first = AdvisorNarrativeLLM(summary="Your returns are guaranteed.", allocation_explanation="x",
                                action_plan=["a"], risk_explanations=[])
    fixed = AdvisorNarrativeLLM(summary="Returns are not guaranteed; review the plan every year.",
                                allocation_explanation="x", action_plan=["a"], risk_explanations=[])
    replies = iter([first, fixed])
    llm = FakeLLM({"AdvisorNarrativeLLM": lambda _u: next(replies)})
    res = run_plan(PROFILE, GOAL, ANSWERS, llm, safety_fn=make_plan_safety_fn(llm, use_llm=False))
    assert res.safety_status == "passed" and res.revision_count == 1 and res.plan.narrative is not None


def test_qa_graph_withholds_unsafe_rag_answer():
    answer = RagAnswerLLM(answer="A SIP will definitely double your money (source: sip.md).",
                          sources_used=["sip.md"], answerable=True)
    llm = FakeLLM({"RouteLLM": RouteLLM(route="knowledge", reason="concept"), "RagAnswerLLM": answer})
    r = ask("What is a SIP?", llm, retriever=KeywordRetriever(), safety_fn=make_qa_safety_fn(llm, use_llm=False))
    assert r.answer.startswith("This answer was withheld")
    assert "high:guaranteed_return" in r.safety_flags


def test_qa_graph_passes_safe_rag_answer():
    answer = RagAnswerLLM(answer="A SIP invests a fixed amount regularly, as low as Rs 500 (source: sip.md). "
                                 "Returns are not guaranteed.", sources_used=["sip.md"], answerable=True)
    llm = FakeLLM({"RouteLLM": RouteLLM(route="knowledge", reason="concept"), "RagAnswerLLM": answer})
    r = ask("What is a SIP?", llm, retriever=KeywordRetriever(), safety_fn=make_qa_safety_fn(llm, use_llm=False))
    assert r.answer.startswith("A SIP invests") and not any(f.startswith("high") for f in r.safety_flags)


# --- Allocation claims must match the calculated allocation (found in the live Gemini run) ---
from agents.safety_agent import check_allocation_claims

MODERATE = {"equity_pct": 60.0, "debt_pct": 30.0, "gold_pct": 10.0, "effective_category": "Moderate"}


def test_wrong_allocation_percentages_are_flagged():
    text = "An illustrative asset allocation of 30.0% equity, 60.0% debt, and 10.0% gold fits you."
    flags = check_allocation_claims(text, MODERATE)
    assert flags and flags[0].category.value == "hallucination" and flags[0].severity.value == "high"


def test_wrong_direction_of_mix_is_flagged():
    assert check_allocation_claims("An allocation leaning toward debt fits your profile.", MODERATE)
    assert check_allocation_claims("Maintain a conservative asset mix for your horizon.", MODERATE)


def test_correct_allocation_text_passes():
    text = ("The illustrative allocation of 60% equity, 30% debt and 10% gold fits your Moderate category. "
            "Market risk: holding 60.0% in equity means ups and downs. Debt at 30% adds stability.")
    assert check_allocation_claims(text, MODERATE) == []


def test_conservative_table_can_lean_toward_debt():
    cons = {"equity_pct": 30.0, "debt_pct": 60.0, "gold_pct": 10.0, "effective_category": "Conservative"}
    assert check_allocation_claims("A heavier weighting in debt with a conservative mix gives stability.", cons) == []
