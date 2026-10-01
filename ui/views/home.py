import streamlit as st

from ui import state
from ui.components import disclaimer
from ui.nav import PAGES


def render():
    st.title("AI Personal Financial Advisor")
    st.subheader("Goal-Based Financial Planning Powered by Multi-Agent AI")
    st.write(
        "Tell the app about your finances, one goal and your comfort with risk. Specialised AI agents then "
        "work out whether your goal is on track, how much you need to invest each month, an illustrative "
        "asset allocation, and a plain language plan. Every number is calculated by Python, not by the AI."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Start Financial Assessment", type="primary", icon=":material/arrow_forward:"):
            st.switch_page(PAGES["profile"])
    with c2:
        users = state.sample_users()
        options = {u["user_id"]: f"{u['user_id']}: {u['goal_type']}, age {u['age']} ({u['note']})" for u in users}
        with st.popover("Load a sample user (demo)", icon=":material/person:"):
            uid = st.selectbox("Synthetic sample users", list(options), format_func=options.get)
            if st.button("Load this user"):
                state.load_sample_user(uid)
                st.success(f"Loaded {uid}. Open any page, or generate the plan from Risk Assessment.")

    st.divider()
    st.markdown("#### How it works")
    cols = st.columns(4)
    steps = [
        (":material/person:", "Profile & Goal Agent", "Understands your cash flow and reads your goal in your own words."),
        (":material/speed:", "Risk Agent", "Scores a 7 question questionnaire (7 to 21) and explains it."),
        (":material/calculate:", "Calculation Agent", "Calls Python calculators for SIP, inflation, gap and health ratios."),
        (":material/donut_large:", "Portfolio Agent", "Picks an illustrative equity / debt / gold split with a time horizon guardrail."),
    ]
    steps2 = [
        (":material/menu_book:", "Knowledge Agent (RAG)", "Answers from SEBI, AMFI and NCFE investor education material."),
        (":material/verified_user:", "Safety Agent", "Blocks promised returns, invented numbers or funds, and risky mismatches."),
        (":material/description:", "Advisor Agent", "Writes the plan narrative from calculated facts only."),
        (":material/chat:", "AI Advisor Q&A", "Ask questions; the AI picks the right calculator or knowledge source."),
    ]
    for row in (steps, steps2):
        cols = st.columns(4)
        for col, (icon, name, text) in zip(cols, row):
            with col.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(text)

    s = state.service_status()
    st.markdown("#### System status")
    c1, c2 = st.columns(2)
    c1.markdown(f"**LLM:** {'Connected (' + s['model'] + ')' if s['llm'] else 'Not configured (calculations only)'}")
    c2.markdown(f"**Knowledge retrieval:** {s['retriever']}")
    if s["retriever_warning"]:
        st.caption(s["retriever_warning"])
    disclaimer()
