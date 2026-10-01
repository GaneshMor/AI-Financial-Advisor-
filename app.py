"""
AI Personal Financial Advisor: Streamlit entry point.

    streamlit run app.py
"""

import streamlit as st

from ui import state
from ui.nav import PAGES
from ui.views import advisor, analysis, goal, goal_analysis, home, portfolio, profile, report, risk, what_if

st.set_page_config(page_title="AI Personal Financial Advisor", page_icon=":material/savings:", layout="wide")
state.init_state()

# Demo mode for presentations: open http://localhost:8501/?demo=U02 to load a synthetic
# sample user and generate the plan once for this browser session.
demo_user = st.query_params.get("demo")
if demo_user and st.session_state.get("demo_loaded") != demo_user:
    if any(u["user_id"] == demo_user for u in state.sample_users()):
        state.load_sample_user(demo_user)
        with st.spinner(f"Loading demo user {demo_user} and generating the plan..."):
            state.generate_plan()
        st.session_state.demo_loaded = demo_user


# st.Page needs a distinct named function per page
def page_home(): home.render()
def page_profile(): profile.render()
def page_goal(): goal.render()
def page_risk(): risk.render()
def page_analysis(): analysis.render()
def page_goal_analysis(): goal_analysis.render()
def page_portfolio(): portfolio.render()
def page_advisor(): advisor.render()
def page_what_if(): what_if.render()
def page_report(): report.render()


PAGES.update({
    "home": st.Page(page_home, title="Home", icon=":material/home:", url_path="home", default=True),
    "profile": st.Page(page_profile, title="User Profile", icon=":material/person:", url_path="profile"),
    "goal": st.Page(page_goal, title="Financial Goal", icon=":material/flag:", url_path="goal"),
    "risk": st.Page(page_risk, title="Risk Assessment", icon=":material/speed:", url_path="risk"),
    "analysis": st.Page(page_analysis, title="Financial Analysis", icon=":material/account_balance_wallet:", url_path="analysis"),
    "goal_analysis": st.Page(page_goal_analysis, title="Goal Analysis", icon=":material/insights:", url_path="goal-analysis"),
    "portfolio": st.Page(page_portfolio, title="Portfolio", icon=":material/donut_large:", url_path="portfolio"),
    "advisor": st.Page(page_advisor, title="AI Advisor", icon=":material/chat:", url_path="advisor"),
    "what_if": st.Page(page_what_if, title="What If?", icon=":material/tune:", url_path="what-if"),
    "report": st.Page(page_report, title="Final Report", icon=":material/description:", url_path="report"),
})

nav = st.navigation({
    "": [PAGES["home"]],
    "Your inputs": [PAGES["profile"], PAGES["goal"], PAGES["risk"]],
    "Your plan": [PAGES["analysis"], PAGES["goal_analysis"], PAGES["portfolio"], PAGES["what_if"]],
    "AI": [PAGES["advisor"], PAGES["report"]],
})

with st.sidebar:
    done = state.steps_done()
    st.progress(sum(done.values()) / 4, text=f"{sum(done.values())} of 4 steps done")
    for key, label in (("profile", "Profile"), ("goal", "Goal"), ("risk", "Risk answers"), ("plan", "Plan generated")):
        st.caption(f"{':material/check_circle:' if done[key] else ':material/radio_button_unchecked:'} {label}")
    s = state.service_status()
    st.caption(f"LLM: {'connected' if s['llm'] else 'off'} · Retrieval: {s['retriever']}")
    if st.button("Reset all inputs", icon=":material/restart_alt:"):
        state.reset_all()
        st.rerun()

nav.run()
