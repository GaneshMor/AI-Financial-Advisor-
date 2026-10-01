import streamlit as st

import config
from schemas.models import GoalType
from tools.inflation import inflation_adjusted_value
from ui.components import disclaimer, page_header
from ui.nav import PAGES
from utils.formatting import format_inr
from utils.validation import validate_inputs

GOALS = [g.value for g in GoalType]
DEFAULT_GOAL = dict(goal_type="House", goal_amount=2500000.0, amount_is_present_value=True,
                    current_goal_savings=0.0, horizon_years=7, description="")


def render():
    page_header("Financial Goal", "One goal per plan in this prototype.")
    d = st.session_state.goal_data or DEFAULT_GOAL
    a = st.session_state.assumptions

    with st.form("goal_form"):
        c1, c2 = st.columns(2)
        goal_type = c1.selectbox("Goal type", GOALS, index=GOALS.index(d["goal_type"]))
        horizon = c2.slider("Time horizon (years)", 1, config.MAX_HORIZON_YEARS, int(d["horizon_years"]))
        c1, c2 = st.columns(2)
        amount = c1.number_input("Goal amount (₹)", min_value=1.0, value=float(d["goal_amount"]), step=50000.0)
        basis = c2.radio("This amount is in", ["Today's value (adjust for inflation)", "Future value (no adjustment)"],
                         index=0 if d["amount_is_present_value"] else 1)
        saved = st.number_input("Savings already set aside for this goal (₹)", min_value=0.0,
                                value=float(d["current_goal_savings"]), step=10000.0)
        desc = st.text_area("Describe the goal in your own words (optional, read by the AI)",
                            value=d.get("description") or "", max_chars=500,
                            placeholder="e.g. Down payment for a 2BHK in Pune in about 7 years")
        with st.expander("Assumptions (illustrative, you can change them)"):
            infl = st.number_input("Annual inflation %", 0.0, config.MAX_INFLATION_PCT, float(a["inflation_pct"]), 0.5)
            override = st.checkbox("Override the expected return (otherwise it comes from your allocation)",
                                   value=a["override_return"])
            ret = st.number_input("Expected annual return %", 0.0, config.MAX_RETURN_PCT, float(a["expected_return_pct"]), 0.5)
            st.caption(config.ASSUMPTION_LABEL)
        submitted = st.form_submit_button("Save and continue", type="primary")

    if submitted:
        goal = dict(goal_type=goal_type, goal_amount=amount, amount_is_present_value=basis.startswith("Today"),
                    current_goal_savings=saved, horizon_years=int(horizon), description=desc.strip() or None)
        profile = st.session_state.profile_data
        # validate the goal (a neutral placeholder profile is used if the profile is not saved yet)
        report = validate_inputs(profile or _placeholder_profile(), goal)
        if report.errors:
            for e in report.errors:
                st.error(e)
        else:
            st.session_state.goal_data = goal
            st.session_state.assumptions = {"inflation_pct": infl, "override_return": override, "expected_return_pct": ret}
            st.success("Goal saved.")
            if profile:
                for w in report.warnings:
                    st.warning(w)

    g = st.session_state.goal_data
    if g:
        if g["amount_is_present_value"]:
            fv = inflation_adjusted_value(g["goal_amount"], st.session_state.assumptions["inflation_pct"], g["horizon_years"])
            st.metric("Inflation adjusted target", format_inr(fv),
                      help=f"{format_inr(g['goal_amount'])} today, at {st.session_state.assumptions['inflation_pct']}% "
                           f"inflation for {g['horizon_years']} years.", border=True)
        if st.button("Next: Risk Assessment", icon=":material/arrow_forward:"):
            st.switch_page(PAGES["risk"])
    disclaimer()


def _placeholder_profile() -> dict:
    return dict(age=30, monthly_income=1, monthly_expenses=0, monthly_debt=0, current_savings=0,
                current_investments=0, monthly_investment=0, dependents=0, emergency_fund=0)
