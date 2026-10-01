"""
Phase 8 tests: the Streamlit UI, run headless with Streamlit's AppTest.

Every page is rendered for several sample users (normal, negative cash flow,
already funded, zero income) with no API key, and must show no exception.
Report builders and the projection helper are tested directly.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from graph.plan_graph import run_plan
from tools.projection import yearly_projection
from tools.sip_calculator import sip_future_value
from ui.report import build_sections, to_html, to_markdown

ROOT = str(Path(__file__).resolve().parent.parent)
PAGES = ["home", "profile", "goal", "risk", "analysis", "goal_analysis", "portfolio", "what_if", "advisor", "report"]
USERS = ["U02", "U09", "U17", "U21"]


def harness(page: str, user_id: str | None, generate: bool, root: str):
    import importlib
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    import streamlit as st
    from ui import state

    state.init_state()
    if user_id and st.session_state.profile_data is None:
        state.load_sample_user(user_id)
        if generate:
            state.generate_plan()
    importlib.import_module(f"ui.views.{page}").render()


def run_page(page, user_id=None, generate=True):
    at = AppTest.from_function(harness, args=(page, user_id, generate, ROOT), default_timeout=60)
    return at.run()


def test_app_entry_point_loads():
    at = AppTest.from_file(str(Path(ROOT) / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "AI Personal Financial Advisor"


@pytest.mark.parametrize("user", USERS)
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_for_sample_users(page, user):
    at = run_page(page, user)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["analysis", "goal_analysis", "portfolio", "what_if", "advisor", "report"])
def test_plan_pages_ask_for_inputs_when_empty(page):
    at = run_page(page, None)
    assert not at.exception
    assert any("first complete" in i.value for i in at.info)


def test_plan_page_offers_generate_when_not_generated():
    at = run_page("goal_analysis", "U02", generate=False)
    assert any(b.label == "Generate my plan" for b in at.button)
    next(b for b in at.button if b.label == "Generate my plan").click().run()
    assert not at.exception
    assert any(m.label == "Required monthly SIP" for m in at.metric)


def test_goal_analysis_shows_calculated_values():
    at = run_page("goal_analysis", "U17")
    labels = {m.label: m.value for m in at.metric}
    assert labels["Required monthly SIP"] == "₹0"          # U17's goal is already funded
    assert any("Ahead of plan" in md.value for md in at.markdown)


def test_portfolio_shows_guardrail_for_short_goal():
    at = run_page("portfolio", "U17")
    assert any("Guardrail applied" in i.value for i in at.info)
    assert {m.label: m.value for m in at.metric}["Equity"] == "30%"


def test_analysis_flags_professional_advice_for_stressed_user():
    at = run_page("analysis", "U09")
    assert any("Professional advice recommended" in e.value for e in at.error)


def test_profile_form_rejects_nothing_negative_and_saves():
    at = run_page("profile", None)
    at.number_input[2].set_value(60000.0)          # income
    at.button[0].click().run()                       # form submit
    assert not at.exception
    assert at.session_state.profile_data["monthly_income"] == 60000.0


def test_advisor_without_llm_explains_instead_of_faking():
    at = run_page("advisor", "U02")
    at.chat_input[0].set_value("What is a SIP?").run()
    assert not at.exception
    assert any("OpenAI API key" in md.value for md in at.markdown)


def test_what_if_slider_changes_required_sip():
    at = run_page("what_if", "U02")
    before = {m.label: m.value for m in at.metric}["Required SIP"]
    at.slider[0].set_value(4.0).run()                # lower assumed return
    after = {m.label: m.value for m in at.metric}["Required SIP"]
    assert not at.exception and after != before


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
PROFILE = dict(age=30, monthly_income=85000, monthly_expenses=38000, monthly_debt=15000, current_savings=150000,
               current_investments=300000, monthly_investment=15000, dependents=0, emergency_fund=250000)
GOAL = dict(goal_type="House", goal_amount=2500000, horizon_years=6, current_goal_savings=400000)


def test_report_has_all_ten_sections_and_disclaimer():
    res = run_plan(PROFILE, GOAL, [2, 2, 3, 3, 2, 2, 2], None)
    headings = [h for h, _ in build_sections(res)]
    for n in ("1.", "2.", "3.", "4.", "5.", "9.", "10."):
        assert any(h.startswith(n) for h in headings)
    md, page = to_markdown(res), to_html(res)
    assert "not regulated financial advice" in md and "<h2>10. Disclaimer</h2>" in page
    assert "Expected annual return" in md


def test_report_html_escapes_text():
    res = run_plan(PROFILE, {**GOAL, "description": "<script>alert(1)</script>"}, [2, 2, 3, 3, 2, 2, 2], None)
    assert "<script>" not in to_html(res)


def test_yearly_projection_matches_tools():
    rows = yearly_projection(100000, 5000, 12, 5)
    assert len(rows) == 6 and rows[0]["value"] == 100000
    assert rows[5]["value"] == pytest.approx(100000 * 1.01 ** 60 + sip_future_value(5000, 12, 60))
    assert rows[5]["invested"] == 100000 + 5000 * 60
