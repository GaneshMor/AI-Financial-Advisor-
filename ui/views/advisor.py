import streamlit as st

from graph.qa_graph import ask
from ui import state
from ui.components import disclaimer, page_header, require_plan
from ui.nav import PAGES

EXAMPLES = [
    "Can I achieve my goal with my current SIP?",
    "Why is my allocation this way?",
    "What happens if inflation increases to 8%?",
    "What happens if my return is only 8%?",
    "How much more should I invest?",
    "What is diversification?",
]
ROUTE_LABEL = {"calculation": "Calculation Agent (tool calling)", "knowledge": "Knowledge Agent (RAG)",
               "out_of_scope": "Out of scope", "no_llm": "No LLM configured", "error": "Router error"}


def _show(result):
    if result.answer:
        st.markdown(result.answer)
    for e in result.errors:
        st.caption(f":material/error: {e}")
    meta = []
    if result.route:
        meta.append(f"Handled by: {ROUTE_LABEL.get(result.route, result.route)}")
    if result.tool_calls:
        meta.append("Tools: " + ", ".join(t.tool for t in result.tool_calls))
    if result.sources_used:
        meta.append("Sources: " + ", ".join(result.sources_used))
    if result.safety_flags:
        meta.append("Safety flags: " + ", ".join(result.safety_flags))
    if meta:
        st.caption(" · ".join(meta))
    if result.tool_calls:
        with st.expander("Tool inputs and outputs"):
            for t in result.tool_calls:
                st.markdown(f"**{t.tool}**")
                st.json({"inputs": t.inputs, "output": t.output}, expanded=False)


def render():
    page_header("AI Advisor", "Ask about your plan or about investing concepts.")
    if not require_plan(PAGES):
        return
    res = state.plan()
    if not state.service_status()["llm"]:
        st.info("The AI Advisor needs an OpenAI API key in the .env file. The other pages work without it.")

    for item in st.session_state.qa_history:
        with st.chat_message("user"):
            st.markdown(item.question)
        with st.chat_message("assistant"):
            _show(item)

    st.caption("Try:")
    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(EXAMPLES):
        if cols[i % 3].button(q, key=f"ex{i}", width="stretch"):
            clicked = q
    question = st.chat_input("Ask a question") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question, state.llm_client(), state.retriever(), res.plan, state.qa_context(res),
                             safety_fn=state.qa_safety_fn(), log=st.session_state.log)
            _show(result)
        st.session_state.qa_history.append(result)
    disclaimer()
