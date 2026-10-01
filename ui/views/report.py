import json

import streamlit as st

from ui import state
from ui.components import page_header, require_plan
from ui.nav import PAGES
from ui.report import SAFETY_LABELS, build_sections, to_html, to_markdown


def render():
    page_header("Final Report", "Your complete plan. Download it as Markdown or as HTML (print to PDF).")
    if not require_plan(PAGES):
        return
    res = state.plan()

    status = res.safety_status
    icon = ":material/verified_user:" if status.startswith("passed") else ":material/gpp_maybe:"
    st.info(f"{icon} Safety check: {SAFETY_LABELS.get(status, status)}"
            + (f" (revised {res.revision_count} time(s))" if res.revision_count else ""))

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download report (.md)", to_markdown(res), "financial_plan.md", "text/markdown",
                       icon=":material/download:", width="stretch")
    c2.download_button("Download report (.html)", to_html(res), "financial_plan.html", "text/html",
                       icon=":material/download:", width="stretch")
    c3.download_button("Download session log (.jsonl)",
                       "\n".join(json.dumps(r) for r in st.session_state.log.to_records()),
                       "session_log.jsonl", "application/json", icon=":material/receipt_long:",
                       width="stretch", help="Agent steps and tool calls. Field names only, no amounts.")

    for heading, items in build_sections(res):
        with st.container(border=True):
            st.markdown(f"#### {heading}")
            rows = [i for i in items if i[0] == "row"]
            if rows:
                st.table({"Item": [r[1] for r in rows], "Value": [str(r[2]) for r in rows]})
            for i in items:
                if i[0] == "li":
                    st.markdown(f"- {i[1]}")
            for i in items:
                if i[0] == "p":
                    st.write(i[1])

    with st.expander("Agent trace (LangGraph)"):
        st.write(" → ".join(res.trace))
        if res.safety_report and res.safety_report.flags:
            st.write("Safety flags on the final version:")
            for f in res.safety_report.flags:
                st.markdown(f"- **{f.severity.value} / {f.category.value}**: {f.text}")
        if res.errors:
            st.write("Non fatal issues:", res.errors)
