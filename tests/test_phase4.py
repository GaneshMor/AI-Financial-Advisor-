"""
Phase 4 tests: allocation rules, LangChain tools, every agent (with the LLM
working, failing, and absent), and the full offline pipeline for all users.

The LLM is replaced by test doubles here (tests/fakes.py). These tests check
the agents' logic, not the quality of real LLM answers.
"""

import csv
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

import config
from agents import llm as llm_module
from agents.advisor_agent import escalation_reasons, run_advisor_agent
from agents.calculation_agent import answer_calculation_question, plan_context_for_qa, run_calculation_agent
from agents.llm import LLMCallError, LLMClient, LLMUnavailableError
from agents.pipeline import run_plan_pipeline
from agents.portfolio_agent import run_portfolio_agent
from agents.profile_agent import run_profile_agent
from agents.rag_agent import NOT_FOUND, run_rag_agent
from agents.risk_agent import run_risk_agent
from schemas.agent_outputs import AdvisorNarrativeLLM, PortfolioExplanationLLM, RagAnswerLLM, RiskExplanationLLM, RiskItemLLM
from schemas.models import GoalInput, ParsedGoal, RetrievedChunk, RiskCategory, ToolName, UserProfile
from schemas.risk_questionnaire import parse_answers
from tests.fakes import FakeLLM, ScriptedChatModel, StubRetriever
from tools.allocation import compute_allocation
from tools.lc_tools import get_lc_tools, tools_by_name
from utils.logging import SessionLog
from utils.validation import split_user_row, validate_inputs

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "data" / "sample_users.csv").open(encoding="utf-8") as f:
    USERS = list(csv.DictReader(f))

PROFILE = UserProfile(age=30, monthly_income=85000, monthly_expenses=38000, monthly_debt=15000,
                      current_savings=150000, current_investments=300000, monthly_investment=15000,
                      dependents=0, emergency_fund=250000)
GOAL = GoalInput(goal_type="House", goal_amount=2500000, horizon_years=6, current_goal_savings=400000,
                 description="Save 25 lakh for a flat down payment in 6 years")
ANSWERS = [2, 2, 3, 3, 2, 2, 2]  # score 16, Moderate


# ---------------------------------------------------------------------------
# Allocation rules (tool 7)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cat, ret", [("Conservative", 8.6), ("Moderate", 10.1), ("Aggressive", 10.85)])
def test_weighted_returns(cat, ret):
    a = compute_allocation(cat, 10, "House", 40000, 0, 500000)
    assert a.weighted_return_pct == pytest.approx(ret)
    assert a.guardrail_applied is False


@pytest.mark.parametrize("cat, years, expected", [
    ("Aggressive", 2, "Conservative"), ("Moderate", 2, "Conservative"), ("Aggressive", 3, "Moderate"),
    ("Aggressive", 5, "Moderate"), ("Aggressive", 6, "Aggressive"), ("Conservative", 1, "Conservative"),
    ("Moderate", 4, "Moderate"),
])
def test_horizon_guardrail(cat, years, expected):
    assert compute_allocation(cat, years, "Vehicle", 1, 0, 0).effective_category.value == expected


def test_emergency_goal_is_all_debt():
    a = compute_allocation("Aggressive", 2, "Emergency Fund", 20000, 0, 0)
    assert (a.equity_pct, a.debt_pct, a.gold_pct) == (0, 100, 0) and a.guardrail_applied


def test_emergency_reserve_gap():
    assert compute_allocation("Moderate", 10, "House", 40000, 10000, 200000).emergency_reserve_gap == 100000
    assert compute_allocation("Moderate", 10, "House", 40000, 10000, 400000).emergency_reserve_gap == 0


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------
def test_tool_names_match_enum():
    assert {t.name for t in get_lc_tools()} == {t.value for t in ToolName}


def test_tool_invocation_and_errors():
    t = tools_by_name()
    assert t["calculate_required_sip"].invoke(
        {"target_amount": 1000000, "annual_return_pct": 12, "horizon_years": 5})["required_monthly_sip"] == 12123.22
    assert t["calculate_inflation"].invoke(
        {"present_value": 1000000, "annual_inflation_pct": 6, "years": 10})["future_cost"] == 1790847.7
    assert "error" in t["calculate_inflation"].invoke({"present_value": -5, "annual_inflation_pct": 6, "years": 10})


def test_scenario_tool_lower_return_needs_more_sip():
    base = dict(goal_amount=1000000, amount_is_present_value=True, horizon_years=10,
                annual_inflation_pct=6, annual_return_pct=10, current_monthly_sip=5000)
    out = tools_by_name()["run_what_if_scenario"].invoke({**base, "new_annual_return_pct": 8})
    assert out["what_if"]["required_monthly_sip"] > out["base"]["required_monthly_sip"]


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def test_no_api_key_raises(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    with pytest.raises(LLMUnavailableError):
        LLMClient()
    assert llm_module.get_llm() is None


def test_llm_errors_are_wrapped():
    client = LLMClient(chat_model=GenericFakeChatModel(messages=iter([])))
    with pytest.raises(LLMCallError):
        client.structured(RiskExplanationLLM, "sys", "user")


# ---------------------------------------------------------------------------
# Profile agent
# ---------------------------------------------------------------------------
def test_profile_agent_detects_discrepancy():
    parsed = ParsedGoal(goal_type="House", goal_amount=4000000, horizon_years=6, priority="High")
    out = run_profile_agent(PROFILE, GOAL, FakeLLM({"ParsedGoal": parsed}))
    assert out.llm_used and out.goal_priority.value == "High"
    assert len(out.discrepancies) == 1 and "40,00,000" in out.discrepancies[0]
    assert out.monthly_surplus == 32000 and out.free_investable_surplus == 17000


def test_profile_agent_no_discrepancy_when_consistent():
    parsed = ParsedGoal(goal_type="House", goal_amount=2500000, horizon_years=6)
    assert run_profile_agent(PROFILE, GOAL, FakeLLM({"ParsedGoal": parsed})).discrepancies == []


@pytest.mark.parametrize("llm", [None, FakeLLM(fail=True)])
def test_profile_agent_fallback(llm):
    out = run_profile_agent(PROFILE, GOAL, llm)
    assert not out.llm_used and out.llm_error and out.parsed_goal is None
    assert out.monthly_surplus == 32000  # numbers still work


# ---------------------------------------------------------------------------
# Risk agent
# ---------------------------------------------------------------------------
def test_risk_agent_llm_cannot_change_score():
    exp = RiskExplanationLLM(reasoning="Moderate because ...", key_risk_factors=["Stable income"])
    fake = FakeLLM({"RiskExplanationLLM": exp})
    out = run_risk_agent(ANSWERS, fake, PROFILE, GOAL)
    assert (out.risk.score, out.risk.category) == (16, RiskCategory.MODERATE)
    assert out.factors_source == "llm" and out.risk.reasoning.startswith("Moderate")
    assert '"score": 16' in fake.calls[0][2]  # score passed to the LLM as a fact


def test_risk_agent_fallback_rule_factors():
    out = run_risk_agent([1, 2, 2, 1, 1, 2, 2], FakeLLM(fail=True))
    assert out.factors_source == "rules" and out.llm_error
    assert "Short investment horizon (under 3 years)" in out.risk.key_risk_factors
    assert len(out.risk.key_risk_factors) == 3


# ---------------------------------------------------------------------------
# Portfolio agent
# ---------------------------------------------------------------------------
def test_portfolio_agent_with_llm():
    exp = PortfolioExplanationLLM(why_selected="w", risk_effect="r", horizon_effect="h", remaining_risks=["x"])
    risk = run_risk_agent(ANSWERS, None).risk
    out = run_portfolio_agent(risk, PROFILE, GOAL, 17000, FakeLLM({"PortfolioExplanationLLM": exp}))
    assert out.allocation.equity_pct == 60 and out.allocation.explanation == "w"


def test_portfolio_agent_guardrail_without_llm():
    risk = run_risk_agent([3] * 7, None).risk
    goal = GOAL.model_copy(update={"horizon_years": 2})
    out = run_portfolio_agent(risk, PROFILE, goal, 17000, None)
    assert out.allocation.effective_category == RiskCategory.CONSERVATIVE and out.llm_error


# ---------------------------------------------------------------------------
# Calculation agent
# ---------------------------------------------------------------------------
def _calc(log=None):
    risk = run_risk_agent(ANSWERS, None).risk
    alloc = run_portfolio_agent(risk, PROFILE, GOAL, 17000, None).allocation
    return run_calculation_agent(PROFILE, GOAL, alloc, log)


def test_calculation_agent_uses_allocation_return_and_records_tools():
    log = SessionLog()
    calc = _calc(log)
    assert calc.assumptions.expected_return_pct == pytest.approx(10.1)
    assert [r.tool for r in calc.tool_calls] == ["financial_health", "plan_goal", "standard_scenarios"]
    assert all(r.success for r in calc.tool_calls)
    assert calc.goal_plan["gap"]["status"] in {"shortfall", "on_track", "ahead"}
    assert len(calc.scenarios) == 3


def test_log_never_contains_values():
    log = SessionLog()
    _calc(log)
    text = json.dumps(log.to_records())
    for value in ("85000", "38000", "2500000", "400000"):
        assert value not in text


def test_qa_tool_calling_loop():
    calc = _calc()
    context = plan_context_for_qa(PROFILE, GOAL, calc)
    call = {"name": "calculate_required_sip", "args": {"target_amount": 1000000, "annual_return_pct": 12,
                                                       "horizon_years": 5}, "id": "call_1"}
    chat = ScriptedChatModel([AIMessage(content="", tool_calls=[call]),
                              AIMessage(content="You would need about Rs 12,123 a month, assuming 12%.")])
    out = answer_calculation_question("How much SIP for 10 lakh in 5 years?", context, FakeLLM(chat_model=chat))
    assert [c.tool for c in out.tool_calls] == ["calculate_required_sip"]
    assert out.tool_calls[0].output["required_monthly_sip"] == 12123.22
    assert "12,123" in out.answer
    # the tool result was sent back to the model
    assert any(isinstance(m, ToolMessage) for m in chat.seen[1])


def test_qa_unknown_tool_and_no_llm():
    chat = ScriptedChatModel([AIMessage(content="", tool_calls=[{"name": "buy_stock", "args": {}, "id": "c"}]),
                              AIMessage(content="Sorry.")])
    out = answer_calculation_question("q", {}, FakeLLM(chat_model=chat))
    assert out.tool_calls[0].success is False
    assert answer_calculation_question("q", {}, None).llm_error


# ---------------------------------------------------------------------------
# RAG agent
# ---------------------------------------------------------------------------
CHUNKS = [RetrievedChunk(text="Inflation reduces purchasing power.", source_file="inflation.md",
                         title="Inflation", score=0.9)]


def test_rag_removes_invented_sources():
    res = RagAnswerLLM(answer="Inflation reduces... (source: inflation.md)",
                       sources_used=["inflation.md", "rbi_report_2024.pdf"], answerable=True)
    out = run_rag_agent("Why does inflation matter?", StubRetriever(CHUNKS), FakeLLM({"RagAnswerLLM": res}))
    assert out.sources_used == ["inflation.md"]
    assert out.invented_sources_removed == ["rbi_report_2024.pdf"]


def test_rag_not_answerable_and_failures():
    res = RagAnswerLLM(answer="x", sources_used=[], answerable=False)
    assert run_rag_agent("q", StubRetriever(CHUNKS), FakeLLM({"RagAnswerLLM": res})).answer == NOT_FOUND
    empty = run_rag_agent("q", StubRetriever([]), FakeLLM())
    assert empty.answer == NOT_FOUND and not empty.answerable
    broken = run_rag_agent("q", StubRetriever(CHUNKS, fail=True), FakeLLM())
    assert "Retrieval failed" in broken.llm_error
    assert "not available" in run_rag_agent("q", None, FakeLLM()).llm_error


# ---------------------------------------------------------------------------
# Advisor agent
# ---------------------------------------------------------------------------
def _narrative(user_text=None):
    return AdvisorNarrativeLLM(
        summary="s", allocation_explanation="a", action_plan=["one", "two", "three"],
        risk_explanations=[RiskItemLLM(name="Market risk", explanation="e")],
        sources_cited=["inflation.md", "made_up.md"],
    )


def test_advisor_filters_sources_and_passes_facts():
    llm = FakeLLM({"AdvisorNarrativeLLM": _narrative})
    res = run_plan_pipeline(PROFILE, GOAL, ANSWERS, None)
    plan = run_advisor_agent(res.profile, res.risk, res.portfolio, res.calculation, CHUNKS, llm,
                             revision_feedback=["Remove guaranteed language"])
    assert plan.sources == ["inflation.md"]
    prompt = llm.calls[0][2]
    assert "₹" in prompt and "Remove guaranteed language" in prompt
    assert "inflation.md" in prompt


def test_advisor_fallback_keeps_all_numbers():
    plan = run_plan_pipeline(PROFILE, GOAL, ANSWERS, None).plan
    assert plan.narrative is None and plan.llm_error
    assert plan.goal_calculation["required_monthly_sip"] > 0
    assert "not regulated financial advice" in plan.disclaimer


def test_escalation_rules():
    row = next(u for u in USERS if u["user_id"] == "U09")
    r = validate_inputs(*split_user_row(row))
    res = run_plan_pipeline(r.profile, r.goal, parse_answers(row["risk_answers"]), None)
    reasons = escalation_reasons(res.profile, res.calculation)
    assert res.plan.professional_advice_recommended and len(reasons) >= 3


# ---------------------------------------------------------------------------
# Full offline pipeline for every sample user (LLM absent AND LLM failing)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", USERS, ids=[u["user_id"] for u in USERS])
@pytest.mark.parametrize("llm_mode", ["none", "failing"])
def test_pipeline_runs_for_every_user(row, llm_mode):
    r = validate_inputs(*split_user_row(row))
    llm = None if llm_mode == "none" else FakeLLM(fail=True)
    res = run_plan_pipeline(r.profile, r.goal, parse_answers(row["risk_answers"]), llm,
                            StubRetriever([], fail=True))
    plan = res.plan
    assert plan.goal_calculation["target_future_value"] > 0
    assert plan.allocation["equity_pct"] + plan.allocation["debt_pct"] + plan.allocation["gold_pct"] == 100
    assert plan.narrative is None and plan.llm_error
    if int(row["goal_horizon_years"]) < 3:
        assert plan.allocation["effective_category"] == "Conservative"
