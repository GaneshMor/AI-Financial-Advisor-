import streamlit as st

from schemas.risk_questionnaire import BANDS, QUESTIONS, score_answers, suggested_answers
from ui import state
from ui.charts import risk_meter
from ui.components import disclaimer, page_header, run_with_spinner
from ui.nav import PAGES


def render():
    page_header("Risk Assessment", "7 questions, 1 to 3 points each. Score 7 to 21.")
    s = st.session_state
    suggested = suggested_answers(s.profile_data, s.goal_data)
    current = s.risk_answers or [suggested.get(i, 2) for i in range(len(QUESTIONS))]

    answers = []
    for i, q in enumerate(QUESTIONS):
        hint = ""
        if i in suggested:
            hint = f"  \n:gray[Suggested from your numbers: **{q['options'][suggested[i] - 1]}**]"
        choice = st.radio(f"**{i + 1}. {q['text']}**{hint}", q["options"], index=current[i] - 1, key=f"risk_q{i}")
        answers.append(q["options"].index(choice) + 1)

    result = score_answers(answers)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(risk_meter(result.score, result.category.value), config={"displayModeBar": False})
    with c2:
        st.markdown("**Scoring bands**")
        for lo, hi, cat in BANDS:
            mark = " ← you" if cat == result.category else ""
            st.markdown(f"- {lo} to {hi}: {cat.value}{mark}")
        st.caption(result.disclaimer)

    c1, c2 = st.columns(2)
    if c1.button("Save answers", width="stretch"):
        s.risk_answers = answers
        st.success("Answers saved.")
    if c2.button("Save and generate my plan", type="primary", icon=":material/auto_awesome:", width="stretch"):
        s.risk_answers = answers
        if not (s.profile_data and s.goal_data):
            st.error("Please complete the User Profile and Financial Goal pages first.")
        else:
            res = run_with_spinner()
            if res.ok:
                st.switch_page(PAGES["analysis"])
            else:
                st.error("Plan failed: " + "; ".join(res.errors))

    if s.plan_result and s.plan_result.ok and s.plan_result.risk.risk.reasoning:
        with st.expander("AI explanation of your risk profile (from the last generated plan)"):
            st.write(s.plan_result.risk.risk.reasoning)
    disclaimer()
