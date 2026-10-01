import pandas as pd
import streamlit as st

from tools.projection import yearly_projection
from ui import state
from ui.charts import projection_lines, sip_comparison
from ui.components import disclaimer, page_header, require_plan
from ui.nav import PAGES
from utils.formatting import format_inr

STATUS_TEXT = {"shortfall": ":red-badge[:material/error: Shortfall]",
               "on_track": ":green-badge[:material/check_circle: On track]",
               "ahead": ":green-badge[:material/check_circle: Ahead of plan]"}


def render():
    page_header("Goal Analysis", "Calculated by Python tools. The AI does not do this maths.")
    if not require_plan(PAGES):
        return
    res = state.plan()
    gp, goal, asm = res.calculation.goal_plan, res.profile.goal, res.calculation.assumptions
    gap = gp["gap"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Goal amount entered", format_inr(goal.goal_amount), border=True,
              help="Today's value" if goal.amount_is_present_value else "Future value")
    c2.metric("Inflation adjusted target", format_inr(gp["target_future_value"]), border=True,
              help=f"At {asm.inflation_pct}% inflation for {goal.horizon_years} years" if goal.amount_is_present_value
              else "Entered as a future value, so not inflated")
    c3.metric("Projected value (current SIP)", format_inr(gp["projected_value_current_plan"]), border=True,
              delta=format_inr(gp["projected_surplus_or_shortfall"]) + " vs target")

    c1, c2, c3 = st.columns(3)
    c1.metric("Required monthly SIP", format_inr(gp["required_monthly_sip"]), border=True)
    c2.metric("Current monthly SIP", format_inr(gap["current_monthly_sip"]), border=True)
    with c3.container(border=True):
        st.metric("Monthly gap", format_inr(max(gap["monthly_gap"], 0)) if gap["status"] == "shortfall"
                  else format_inr(abs(gap["monthly_gap"])) + " spare")
        st.markdown(STATUS_TEXT[gap["status"]])

    if gap["status"] == "shortfall":
        if gap["gap_affordable"]:
            st.success(f"Your free surplus ({format_inr(gap['free_investable_surplus'])}/month) can cover the gap.")
        else:
            st.error(f"The gap is larger than your free surplus by {format_inr(gap['uncovered_gap'])} a month. "
                     "Options: a longer horizon, a smaller target, or more income.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Required vs current SIP")
        st.plotly_chart(sip_comparison(gp["required_monthly_sip"], gap["current_monthly_sip"]),
                        config={"displayModeBar": False})
    with c2:
        st.markdown("#### Projected growth")
        r = asm.expected_return_pct
        series = {
            "Current SIP": yearly_projection(goal.current_goal_savings, gap["current_monthly_sip"], r, goal.horizon_years),
            "Required SIP": yearly_projection(goal.current_goal_savings, gp["required_monthly_sip"], r, goal.horizon_years),
        }
        st.plotly_chart(projection_lines(series, gp["target_future_value"]), config={"displayModeBar": False})
        with st.expander("Table view"):
            st.dataframe(pd.DataFrame({
                "Year": [x["year"] for x in series["Current SIP"]],
                "Current SIP": [format_inr(x["value"]) for x in series["Current SIP"]],
                "Required SIP": [format_inr(x["value"]) for x in series["Required SIP"]],
            }), hide_index=True)

    st.caption(f"Assumptions: {asm.expected_return_pct}% annual return, {asm.inflation_pct}% inflation, "
               f"{asm.contribution_frequency.lower()}, monthly compounding. {asm.label}")

    with st.expander(":material/build: Tool calls made by the Calculation Agent"):
        for call in res.calculation.tool_calls:
            st.markdown(f"**{call.tool}**: {'success' if call.success else 'error: ' + str(call.error)}")
    disclaimer()
