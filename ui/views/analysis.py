import streamlit as st

from ui import state
from ui.charts import cash_flow_bars
from ui.components import disclaimer, llm_notice, page_header, require_plan, status_badge
from ui.nav import PAGES
from utils.formatting import format_inr


def render():
    page_header("Financial Analysis", "Where your money goes each month.")
    if not require_plan(PAGES):
        return
    res = state.plan()
    h = res.calculation.health

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly income", format_inr(h["monthly_income"]), border=True)
    c2.metric("Monthly expenses + EMIs", format_inr(h["monthly_expenses"] + h["monthly_debt"]), border=True)
    c3.metric("Monthly surplus", format_inr(h["monthly_surplus"]), border=True,
              help="Income minus living expenses minus EMIs.")

    c1, c2, c3 = st.columns(3)
    for col, key in zip((c1, c2, c3), ("savings_rate", "debt_to_income", "emergency_fund_coverage")):
        ind = h[key]
        value = "n/a" if ind["value"] is None else (f"{ind['value']:.1f}%" if ind["unit"] == "%" else f"{ind['value']:.1f} months")
        with col.container(border=True):
            st.metric(ind["name"], value, help=ind["note"])
            st.markdown(status_badge(ind["status"]))

    st.markdown("#### Monthly cash flow")
    st.plotly_chart(cash_flow_bars(h["monthly_income"], h["monthly_expenses"], h["monthly_debt"],
                                   h["monthly_investment"], h["free_investable_surplus"]),
                    config={"displayModeBar": False})
    st.caption(f"Free surplus after your current SIP: **{format_inr(h['free_investable_surplus'])}** a month.")

    for w in res.plan.warnings:
        st.warning(w)
    if res.plan.professional_advice_recommended:
        st.error("**Professional advice recommended.** " + " ".join(res.plan.escalation_reasons)
                 + " Please consider a SEBI registered investment adviser.", icon=":material/support_agent:")
    st.caption(h["disclaimer"])
    llm_notice()
    disclaimer()
