import pandas as pd
import streamlit as st

import config
from tools.goal_gap import GoalPlanInputs
from tools.projection import yearly_projection
from tools.scenario import SCENARIO_LABEL, run_scenario, standard_scenarios
from ui import state
from ui.charts import projection_lines
from ui.components import disclaimer, page_header, require_plan
from ui.nav import PAGES
from utils.formatting import format_inr


def render():
    page_header("What If?", SCENARIO_LABEL)
    if not require_plan(PAGES):
        return
    res = state.plan()
    goal, asm, gp = res.profile.goal, res.calculation.assumptions, res.calculation.goal_plan
    base = GoalPlanInputs(
        goal_amount=goal.goal_amount, amount_is_present_value=goal.amount_is_present_value,
        horizon_years=goal.horizon_years, annual_inflation_pct=asm.inflation_pct,
        annual_return_pct=asm.expected_return_pct, current_goal_savings=goal.current_goal_savings,
        current_monthly_sip=gp["gap"]["current_monthly_sip"], free_investable_surplus=gp["gap"]["free_investable_surplus"],
    )

    c1, c2 = st.columns(2)
    ret = c1.slider("Expected annual return %", 0.0, 20.0, float(round(base.annual_return_pct, 1)), 0.5)
    infl = c2.slider("Inflation %", 0.0, 15.0, float(base.annual_inflation_pct), 0.5)
    c1, c2 = st.columns(2)
    sip = c1.number_input("Monthly SIP (₹)", 0.0, value=float(base.current_monthly_sip), step=1000.0)
    years = c2.slider("Time horizon (years)", 1, config.MAX_HORIZON_YEARS, int(base.horizon_years))

    now = run_scenario(base, "Current plan")
    what = run_scenario(base, "What if", annual_return_pct=ret, annual_inflation_pct=infl,
                        current_monthly_sip=sip, horizon_years=years)

    def change(key):  # no delta arrow when nothing changed
        d = what[key] - now[key]
        return format_inr(d) + " vs current" if abs(d) >= 1 else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Target", format_inr(what["target_future_value"]), border=True,
              delta=change("target_future_value"), delta_color="inverse")
    c2.metric("Required SIP", format_inr(what["required_monthly_sip"]), border=True,
              delta=change("required_monthly_sip"), delta_color="inverse")
    c3.metric("Projected with this SIP", format_inr(what["projected_value_current_plan"]), border=True,
              delta=format_inr(what["projected_surplus_or_shortfall"]) + " vs target")

    series = {
        "Current plan": yearly_projection(goal.current_goal_savings, base.current_monthly_sip, base.annual_return_pct, base.horizon_years),
        "What if": yearly_projection(goal.current_goal_savings, sip, ret, years),
    }
    st.plotly_chart(projection_lines(series, what["target_future_value"]), config={"displayModeBar": False})
    st.caption("Dotted line: target under the What If inflation and horizon.")

    st.markdown("#### Standard scenarios (return ± 2 points)")
    rows = standard_scenarios(base)
    st.dataframe(pd.DataFrame([{
        "Scenario": r["scenario"], "Return %": r["annual_return_pct"], "Required SIP": format_inr(r["required_monthly_sip"]),
        "Projected (current SIP)": format_inr(r["projected_value_current_plan"]),
        "Surplus / shortfall": format_inr(r["projected_surplus_or_shortfall"]), "Status": r["status"],
    } for r in rows]), hide_index=True, width="stretch")
    st.warning(f":material/science: {SCENARIO_LABEL} These numbers are not predictions of market returns.")
    disclaimer()
