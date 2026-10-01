"""
Phase 9 tests: the scoring functions, the offline evaluation run, and the full
evaluation MECHANICS with a test double LLM (the numbers from the fake run are
meaningless; only real runs produce reportable metrics).
"""

import json
import re

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import evaluation.evaluate as ev
from evaluation.metrics import (
    Metric, groundedness, hit_at_k, score_calculation, score_goal_parse, score_safety_case, score_tool_selection,
    summary_table,
)
from schemas.agent_outputs import (
    AdvisorNarrativeLLM, ClaimJudgementLLM, GroundednessLLM, PortfolioExplanationLLM, RagAnswerLLM,
    RiskExplanationLLM, RiskItemLLM, RouteLLM, SafetyJudgeLLM,
)
from schemas.models import ParsedGoal
from tests.fakes import FakeLLM


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def test_metric_display_and_rate():
    m = Metric("x")
    for ok in (True, True, False):
        m.add(ok)
    assert m.display() == "66.7% (2/3)" and m.rate == pytest.approx(2 / 3)
    assert Metric("y", status="not_run").display() == "Not run"
    assert "| x | 66.7% (2/3)" in summary_table([m])


def test_score_calculation_tolerance_and_status():
    res = {"target_future_value": 100.4, "fv_current_goal_savings": 0, "required_monthly_sip": 10,
           "projected_value_current_plan": 5, "gap": {"status": "shortfall"}}
    exp = {"target_future_value": 100, "fv_current_goal_savings": 0, "required_monthly_sip": 10,
           "projected_value_current_plan": 5, "status": "shortfall"}
    assert score_calculation(res, exp)[0]
    assert not score_calculation({**res, "required_monthly_sip": 12}, exp)[0]
    assert not score_calculation(res, {**exp, "status": "ahead"})[0]


def test_score_goal_parse_skips_unstated_fields():
    exp = {"goal_type": "House", "goal_amount": 8000000, "horizon_years": 7, "amount_is_present_value": None}
    ok, fields = score_goal_parse({"goal_type": "House", "goal_amount": 8000000.0, "horizon_years": 7,
                                   "amount_is_present_value": True}, exp)
    assert ok and "amount_is_present_value" not in fields
    assert not score_goal_parse({"goal_type": "House", "goal_amount": 800000, "horizon_years": 7}, exp)[0]


def test_score_goal_parse_missing_fields():
    exp = {"goal_type": "Wealth Creation", "goal_amount": None, "horizon_years": None,
           "missing_fields": ["goal_amount", "horizon_years"]}
    good = {"goal_type": "Wealth Creation", "goal_amount": None, "horizon_years": None,
            "missing_fields": ["goal_amount", "horizon_years"]}
    assert score_goal_parse(good, exp)[0]
    assert not score_goal_parse({**good, "goal_amount": 1000000}, exp)[0]   # invented a number


def test_other_scorers():
    assert hit_at_k(["a.md", "b.md"], "b.md", 2) and not hit_at_k(["a.md", "b.md"], "b.md", 1)
    assert score_tool_selection(["x", "y"], "y") == (True, False)
    assert score_tool_selection([], "y") == (False, False)
    assert score_safety_case({"hallucination"}, set(), high_detected=True) == (False, {"hallucination"})
    assert score_safety_case(set(), set(), high_detected=False)[0]
    assert groundedness([{"supported": True}, {"supported": False}]) == (1, 2)


# ---------------------------------------------------------------------------
# Offline run (real numbers, no API key)
# ---------------------------------------------------------------------------
def test_offline_run_writes_results(tmp_path):
    out = ev.main(["--offline", "--users", "4", "--out", str(tmp_path)])
    data = json.loads((out / "metrics.json").read_text())
    rows = {r["metric"]: r for r in data["metrics"]}
    assert data["config"]["mode"] == "offline"
    assert rows["Calculation accuracy"]["passed"] == rows["Calculation accuracy"]["total"] == 10
    assert rows["Tool selection accuracy"]["status"] == "not_run"
    assert rows["Hallucination rate (unsupported claims)"]["status"] == "not_run"
    assert (out / "summary.md").exists() and (out / "details.csv").exists()


# ---------------------------------------------------------------------------
# Full run mechanics with a test double
# ---------------------------------------------------------------------------
class ToolChat:
    """Calls calculate_required_sip once, then answers."""

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="About Rs 12,123 a month, assuming an illustrative 12% return.")
        return AIMessage(content="", tool_calls=[{"name": "calculate_required_sip", "id": "c1",
                                                  "args": {"target_amount": 1000000, "annual_return_pct": 12,
                                                           "horizon_years": 5}}])


def _risk_expl(user):
    cat = re.search(r'"category": "(\w+)"', user).group(1)
    return RiskExplanationLLM(reasoning=f"Your profile is {cat}.", key_risk_factors=["x"])


def test_full_run_mechanics_with_fake_llm(tmp_path, monkeypatch):
    fake = FakeLLM({
        "ParsedGoal": ParsedGoal(goal_type="House", goal_amount=8000000, horizon_years=7),
        "RiskExplanationLLM": _risk_expl,
        "PortfolioExplanationLLM": PortfolioExplanationLLM(why_selected="w", risk_effect="r", horizon_effect="h",
                                                           remaining_risks=["x"]),
        "AdvisorNarrativeLLM": AdvisorNarrativeLLM(summary="Returns are not guaranteed.", allocation_explanation="a",
                                                   action_plan=["Review yearly."], risk_explanations=[
                                                       RiskItemLLM(name="Market risk", explanation="Values can fall.")]),
        "RouteLLM": RouteLLM(route="knowledge", reason="r"),
        "RagAnswerLLM": RagAnswerLLM(answer="A SIP invests a fixed amount (source: sip.md).",
                                     sources_used=["sip.md"], answerable=True),
        "SafetyJudgeLLM": SafetyJudgeLLM(issues=[]),
        "GroundednessLLM": GroundednessLLM(claims=[ClaimJudgementLLM(claim="c1", supported=True),
                                                   ClaimJudgementLLM(claim="c2", supported=False)]),
    }, chat_model=ToolChat())
    monkeypatch.setattr(ev, "get_llm", lambda: fake)
    monkeypatch.setattr(ev, "LLMClient", lambda **kw: fake)

    out = ev.main(["--users", "3", "--repeats", "2", "--out", str(tmp_path)])
    rows = {r["metric"]: r for r in json.loads((out / "metrics.json").read_text())["metrics"]}
    assert all(r["status"] == "run" for r in rows.values())
    assert rows["Tool selection accuracy"]["total"] == 8
    assert rows["Goal classification accuracy (goal type)"]["total"] == 31
    assert rows["Q&A routing accuracy"]["total"] == 8
    assert rows["Risk explanation consistency (LLM)"]["passed"] == 5
    assert rows["Hallucination rate (unsupported claims)"]["total"] > 0
    assert (out / "manual_review.csv").exists()
