"""
Phase 5 tests: LangGraph orchestration.

Checks node order, fatal vs non fatal failures, the safety revision loop
(with a test safety function, since the real Safety Agent is Phase 7),
and every route of the Q&A graph.
"""

import pytest
from langchain_core.messages import AIMessage

import graph.plan_graph as plan_graph_module
from graph.plan_graph import MAX_REVISIONS, run_plan
from graph.qa_graph import NO_LLM_ANSWER, PLAN_SOURCE, ask, plan_facts_chunk
from prompts.agent_prompts import OUT_OF_SCOPE_ANSWER
from schemas.agent_outputs import AdvisorNarrativeLLM, RagAnswerLLM, RiskItemLLM, RouteLLM
from schemas.models import RetrievedChunk, SafetyFlag, SafetyReport
from tests.fakes import FakeLLM, ScriptedChatModel, StubRetriever

PROFILE = dict(age=30, monthly_income=85000, monthly_expenses=38000, monthly_debt=15000,
               current_savings=150000, current_investments=300000, monthly_investment=15000,
               dependents=0, emergency_fund=250000)
GOAL = dict(goal_type="House", goal_amount=2500000, horizon_years=6, current_goal_savings=400000)
ANSWERS = [2, 2, 3, 3, 2, 2, 2]
FULL_CHAIN = ["validate", "profile", "risk", "portfolio", "calculation", "knowledge", "advisor"]

HIGH = SafetyReport(flags=[SafetyFlag(category="guaranteed_return", severity="high",
                                      text="will definitely return 15%", suggestion="use assumption language")])
CLEAN = SafetyReport()


def narrative(_user=None):
    return AdvisorNarrativeLLM(summary="s", allocation_explanation="a", action_plan=["1", "2", "3"],
                               risk_explanations=[RiskItemLLM(name="Market risk", explanation="e")])


# ---------------------------------------------------------------------------
# Plan graph
# ---------------------------------------------------------------------------
def test_plan_graph_order_without_safety():
    res = run_plan(PROFILE, GOAL, ANSWERS, None)
    assert res.ok and res.plan is not None
    assert res.trace == FULL_CHAIN + ["finalize"]
    assert res.safety_status == "not_run"


def test_invalid_input_stops_at_validate():
    res = run_plan({**PROFILE, "monthly_income": -5}, GOAL, ANSWERS, None)
    assert not res.ok and res.plan is None
    assert res.trace == ["validate (invalid input)"]
    assert any("Monthly income" in e for e in res.errors)


def test_bad_risk_answers_are_fatal_but_do_not_crash():
    res = run_plan(PROFILE, GOAL, [1, 2, 3], None)
    assert not res.ok and res.trace[-1] == "risk (failed)"


def test_unexpected_node_error_is_caught(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(plan_graph_module, "run_calculation_agent", boom)
    res = run_plan(PROFILE, GOAL, ANSWERS, None)
    assert not res.ok and "calculation: RuntimeError: disk full" in res.errors


def test_retrieval_failure_is_not_fatal():
    res = run_plan(PROFILE, GOAL, ANSWERS, None, retriever=StubRetriever([], fail=True))
    assert res.ok
    assert any("Retrieval failed" in e for e in res.errors)


def test_safety_loop_stops_after_max_revisions():
    calls = []

    def always_high(plan, state):
        calls.append(1)
        return HIGH

    res = run_plan(PROFILE, GOAL, ANSWERS, FakeLLM({"AdvisorNarrativeLLM": narrative}), safety_fn=always_high)
    assert res.ok
    assert res.revision_count == MAX_REVISIONS
    assert len(calls) == MAX_REVISIONS + 1
    assert res.safety_status == "failed_after_revisions"
    assert res.trace.count("advisor") == MAX_REVISIONS + 1


def test_safety_feedback_reaches_advisor_and_passes():
    llm = FakeLLM({"AdvisorNarrativeLLM": narrative})
    reports = iter([HIGH, CLEAN])
    res = run_plan(PROFILE, GOAL, ANSWERS, llm, safety_fn=lambda p, s: next(reports))
    assert res.safety_status == "passed" and res.revision_count == 1
    advisor_prompts = [u for name, _, u in llm.calls if name == "AdvisorNarrativeLLM"]
    assert len(advisor_prompts) == 2
    assert "(none)" in advisor_prompts[0]
    assert "will definitely return 15%" in advisor_prompts[1]


def test_safety_function_crash_keeps_plan():
    def crash(plan, state):
        raise ValueError("bad regex")
    res = run_plan(PROFILE, GOAL, ANSWERS, None, safety_fn=crash)
    assert res.ok and res.safety_status == "error"


# ---------------------------------------------------------------------------
# Q&A graph
# ---------------------------------------------------------------------------
def _plan():
    return run_plan(PROFILE, GOAL, ANSWERS, None)


def test_qa_without_llm():
    r = ask("What is a SIP?", None)
    assert r.answer == NO_LLM_ANSWER and r.route == "no_llm"


def test_qa_empty_question():
    assert ask("   ", None).errors == ["Please type a question."]


def test_qa_calculation_route_uses_tools():
    chat = ScriptedChatModel([
        AIMessage(content="", tool_calls=[{"name": "calculate_inflation", "id": "c1",
                                           "args": {"present_value": 1000000, "annual_inflation_pct": 6, "years": 5}}]),
        AIMessage(content="It may cost about Rs 13,38,226, assuming 6% inflation."),
    ])
    llm = FakeLLM({"RouteLLM": RouteLLM(route="calculation", reason="needs a number")}, chat_model=chat)
    r = ask("What will 10 lakh cost in 5 years?", llm)
    assert r.route == "calculation" and r.trace == ["router -> calculation", "calculation_agent"]
    assert r.tool_calls[0].tool == "calculate_inflation"
    assert r.tool_calls[0].output["future_cost"] == pytest.approx(1338225.58, abs=0.01)


def test_qa_knowledge_route_can_cite_users_plan():
    res = _plan()
    answer = RagAnswerLLM(answer="Your allocation is Moderate because... (source: your_plan)",
                          sources_used=[PLAN_SOURCE, "made_up.pdf"], answerable=True)
    llm = FakeLLM({"RouteLLM": RouteLLM(route="knowledge", reason="why question"), "RagAnswerLLM": answer})
    r = ask("Why is my allocation moderate?", llm, retriever=None, plan=res.plan)
    assert r.route == "knowledge"
    assert r.sources_used == [PLAN_SOURCE] and r.invented_sources_removed == ["made_up.pdf"]
    rag_prompt = [u for name, _, u in llm.calls if name == "RagAnswerLLM"][0]
    assert "Moderate" in rag_prompt and "equity 60%" in rag_prompt


def test_plan_chunk_contents():
    chunk = plan_facts_chunk(_plan().plan)
    assert chunk.source_file == PLAN_SOURCE and "Required monthly SIP" in chunk.text


def test_qa_out_of_scope():
    llm = FakeLLM({"RouteLLM": RouteLLM(route="out_of_scope", reason="stock tip")})
    r = ask("Which stock will double next month?", llm)
    assert r.answer == OUT_OF_SCOPE_ANSWER and r.tool_calls == []


def test_qa_router_failure():
    r = ask("What is a SIP?", FakeLLM(fail=True))
    assert r.route == "error" and r.answer is None and r.errors


def test_qa_safety_withholds_unsafe_answer():
    chunks = [RetrievedChunk(text="SIP basics", source_file="sip.md", title="SIP", score=0.9)]
    answer = RagAnswerLLM(answer="SIPs always give 15%.", sources_used=["sip.md"], answerable=True)
    llm = FakeLLM({"RouteLLM": RouteLLM(route="knowledge", reason="concept"), "RagAnswerLLM": answer})
    r = ask("What is a SIP?", llm, retriever=StubRetriever(chunks), safety_fn=lambda text, s: HIGH)
    assert r.answer.startswith("This answer was withheld")
    assert r.safety_flags == ["high:guaranteed_return"]
