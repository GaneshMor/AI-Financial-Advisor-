"""Small reusable Streamlit pieces."""

import streamlit as st

from prompts.agent_prompts import DISCLAIMER
from ui import state

STATUS_ICON = {"good": ":material/check_circle:", "warning": ":material/warning:", "critical": ":material/error:"}
HEALTH_STATUS = {
    "Healthy": "good", "Comfortable": "good", "Adequate": "good",
    "Moderate": "warning", "Elevated": "warning", "Partial": "warning",
    "Low": "critical", "High": "critical",
}


def status_badge(status: str) -> str:
    """Status always shown as icon + word, never colour alone."""
    level = HEALTH_STATUS.get(status)
    if level is None:
        return status
    color = {"good": "green", "warning": "orange", "critical": "red"}[level]
    return f":{color}-badge[{STATUS_ICON[level]} {status}]"


def disclaimer() -> None:
    st.caption(f":material/info: {DISCLAIMER}")


def page_header(title: str, subtitle: str | None = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def require_plan(pages: dict) -> bool:
    """Show what is missing and a Generate button. Returns True when a current plan exists."""
    done = state.steps_done()
    if not state.inputs_complete():
        missing = [name for key, name in (("profile", "User Profile"), ("goal", "Financial Goal"),
                                          ("risk", "Risk Assessment")) if not done[key]]
        st.info("To see this page, first complete: " + ", ".join(missing) + ".")
        first = {"User Profile": "profile", "Financial Goal": "goal", "Risk Assessment": "risk"}[missing[0]]
        if st.button(f"Go to {missing[0]}", type="primary"):
            st.switch_page(pages[first])
        return False
    if state.plan() is not None and not state.plan_is_current():
        st.warning("Your inputs changed since the plan was generated. Regenerate to update every page.")
    if state.plan() is None or not state.plan_is_current():
        if st.button("Generate my plan", type="primary", icon=":material/auto_awesome:"):
            run_with_spinner()
            st.rerun()
        if state.plan() is None:
            return False
    res = state.plan()
    if not res.ok:
        st.error("The plan could not be generated:\n\n" + "\n".join(f"- {e}" for e in res.errors))
        return False
    return True


def run_with_spinner():
    status = state.service_status()
    msg = "Running the agents (profile, risk, portfolio, calculation, knowledge, advisor, safety)..."
    if not status["llm"]:
        msg = "Running the calculations (AI explanations are off: no API key)..."
    with st.spinner(msg):
        return state.generate_plan()


def llm_notice() -> None:
    if not state.service_status()["llm"]:
        st.info(":material/key_off: AI explanations are off because no OpenAI API key is set. "
                "All calculations, charts and the allocation still work.")
