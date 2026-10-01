import pandas as pd
import streamlit as st

import config
from ui import state
from ui.charts import allocation_donut
from ui.components import disclaimer, llm_notice, page_header, require_plan
from ui.nav import PAGES
from utils.formatting import format_inr


def render():
    page_header("Portfolio", "Illustrative allocation of your monthly investment. Not a product recommendation.")
    if not require_plan(PAGES):
        return
    res = state.plan()
    a = res.portfolio.allocation
    exp = res.portfolio.explanation

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(allocation_donut(a.equity_pct, a.debt_pct, a.gold_pct), config={"displayModeBar": False})
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Equity", f"{a.equity_pct:.0f}%", border=True)
        m2.metric("Debt", f"{a.debt_pct:.0f}%", border=True)
        m3.metric("Gold", f"{a.gold_pct:.0f}%", border=True)
        st.markdown(f"**Questionnaire category:** {a.risk_category.value}  \n"
                    f"**Effective category used:** {a.effective_category.value}  \n"
                    f"**Weighted assumed return:** {a.weighted_return_pct}% a year")
        if a.guardrail_applied:
            st.info(f":material/shield: **Guardrail applied.** {a.guardrail_reason}")
        if a.emergency_reserve_gap > 0:
            st.warning(f"Build your emergency reserve first: about {format_inr(a.emergency_reserve_gap)} more is "
                       f"needed for {config.EMERGENCY_RESERVE_MONTHS} months of expenses and EMIs.")

    st.markdown("#### Why this allocation?")
    if exp:
        st.write(exp.why_selected)
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown("**How risk affects it**")
            st.write(exp.risk_effect)
        with c2.container(border=True):
            st.markdown("**How time horizon affects it**")
            st.write(exp.horizon_effect)
        st.markdown("**Risks that remain**")
        for r in exp.remaining_risks:
            st.markdown(f"- {r}")
        st.caption("Explanation written by the Portfolio Agent (AI). Percentages decided by fixed rules.")
    else:
        st.write(f"The {a.effective_category.value} table was used because of your questionnaire result"
                 + (" and the time horizon guardrail." if a.guardrail_applied else "."))
        llm_notice()

    with st.expander("All illustrative allocation tables and assumed returns"):
        st.dataframe(pd.DataFrame(config.ALLOCATION_TABLES).T.rename(columns=str.title), width="stretch")
        st.write("Assumed annual returns: " + ", ".join(f"{k} {v}%" for k, v in config.ASSUMED_ANNUAL_RETURNS_PCT.items()))
        st.caption(config.ASSUMPTION_LABEL)
    st.caption(a.label)
    disclaimer()
