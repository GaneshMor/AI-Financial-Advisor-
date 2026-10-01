"""
Final report builder (pure Python, no Streamlit) -> Markdown and printable HTML.

The report follows the 10 sections of the plan. Numbers come from the
deterministic agents; the narrative sections are included only if the AI text
passed the safety check. Every assumption and the disclaimer are always shown.
"""

import html
from datetime import datetime

from graph.plan_graph import PlanRunResult
from utils.formatting import format_inr

SAFETY_LABELS = {
    "passed": "Passed: AI text checked by the Safety Agent",
    "passed_structure_only": "Numbers and allocation checked; no AI text was generated",
    "failed_after_revisions": "AI text withheld after failing safety checks",
    "not_run": "Safety check not run",
    "error": "Safety check could not run",
}


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def build_sections(res: PlanRunResult) -> list[tuple[str, list]]:
    """[(heading, [("row", label, value) | ("p", text) | ("li", text)])]"""
    p = res.plan
    s, g, r, gc, a, asm = p.snapshot, p.goal, p.risk, p.goal_calculation, p.allocation, p.assumptions
    n = p.narrative
    status_word = {"shortfall": "Shortfall", "on_track": "On track", "ahead": "Ahead of plan"}[gc["status"]]
    ef = s["emergency_fund_months"]

    sections: list[tuple[str, list]] = [
        ("1. Financial Snapshot", [
            ("row", "Monthly income", format_inr(s["monthly_income"])),
            ("row", "Monthly living expenses", format_inr(s["monthly_expenses"])),
            ("row", "Monthly EMIs", format_inr(s["monthly_debt"])),
            ("row", "Monthly surplus", format_inr(s["monthly_surplus"])),
            ("row", "Current monthly investment", format_inr(s["current_monthly_investment"])),
            ("row", "Free investable surplus", format_inr(s["free_investable_surplus"])),
            ("row", "Savings rate", _pct(s["savings_rate_pct"])),
            ("row", "Debt to income", _pct(s["debt_to_income_pct"])),
            ("row", "Emergency fund coverage", "n/a" if ef is None else f"{ef:.1f} months"),
        ]),
        ("2. Goal", [
            ("row", "Goal", g["goal_type"]),
            ("row", "Amount entered", format_inr(g["goal_amount_entered"])
             + (" (today's value)" if g["amount_is_present_value"] else " (future value)")),
            ("row", "Time horizon", f"{g['horizon_years']} years"),
            ("row", "Inflation adjusted target", format_inr(g["inflation_adjusted_target"])),
            ("row", "Savings already set aside", format_inr(g["current_goal_savings"])),
        ] + ([("row", "Goal priority (AI assessed)", g["priority"])] if g.get("priority") else [])),
        ("3. Risk Profile", [
            ("row", "Risk score", f"{r['score']} / 21"),
            ("row", "Risk category", r["category"]),
        ] + ([("p", r["reasoning"])] if r.get("reasoning") else [])
          + [("li", f) for f in r["key_risk_factors"]]
          + [("p", r["disclaimer"])]),
        ("4. Goal Calculation", [
            ("row", "Required monthly SIP", format_inr(gc["required_monthly_sip"])),
            ("row", "Current monthly SIP", format_inr(gc["current_monthly_sip"])),
            ("row", "Monthly gap", format_inr(gc["monthly_gap"])),
            ("row", "Status", status_word),
            ("row", "Gap affordable from free surplus", "Yes" if gc["gap_affordable_from_surplus"] else "No"),
            ("row", "Projected value with current SIP", format_inr(gc["projected_value_current_plan"])),
        ]),
        ("5. Illustrative Allocation", [
            ("row", "Equity", f"{a['equity_pct']:.0f}%"),
            ("row", "Debt", f"{a['debt_pct']:.0f}%"),
            ("row", "Gold", f"{a['gold_pct']:.0f}%"),
            ("row", "Effective category", a["effective_category"]),
            ("row", "Weighted assumed return", f"{a['weighted_assumed_return_pct']}%"),
        ] + ([("row", "Emergency reserve still needed", format_inr(a["emergency_reserve_gap"]))]
             if a["emergency_reserve_gap"] > 0 else [])
          + ([("p", f"Guardrail: {a['guardrail_reason']}")] if a["guardrail_applied"] else [])
          + [("p", a["label"])]),
    ]

    if n:
        sections += [
            ("6. Explanation", [("p", n.summary), ("p", n.allocation_explanation)]),
            ("7. Action Plan", [("li", x) for x in n.action_plan]),
            ("8. Risks", [("li", f"{x.name}: {x.explanation}") for x in n.risk_explanations]),
        ]
    else:
        msg = p.llm_error or "AI explanation unavailable."
        sections += [("6-8. Explanation, Action Plan, Risks", [("p", f"Not available: {msg}")])]

    if p.professional_advice_recommended:
        sections.append(("Professional advice recommended", [("li", x) for x in p.escalation_reasons]))
    if p.warnings:
        sections.append(("Warnings", [("li", w) for w in p.warnings]))

    sections += [
        ("9. Assumptions", [
            ("row", "Expected annual return", f"{asm['expected_annual_return_pct']}%"),
            ("row", "Inflation", f"{asm['annual_inflation_pct']}%"),
            ("row", "Asset return assumptions", ", ".join(f"{k} {v}%" for k, v in asm["asset_returns_pct"].items())),
            ("row", "Investment horizon", f"{asm['horizon_years']} years"),
            ("row", "Contribution frequency", asm["contribution_frequency"]),
            ("row", "Compounding", asm["compounding"]),
            ("p", asm["label"]),
        ]),
        ("10. Disclaimer", [("p", p.disclaimer),
                            ("p", f"Safety check: {SAFETY_LABELS.get(res.safety_status, res.safety_status)}.")]
         + ([("p", "Knowledge sources used: " + ", ".join(p.sources))] if p.sources else [])),
    ]
    return sections


def to_markdown(res: PlanRunResult) -> str:
    lines = ["# Personal Financial Plan", f"_Generated {datetime.now():%d %b %Y, %H:%M} by the AI Personal "
             "Financial Advisor (academic prototype)_", ""]
    for heading, items in build_sections(res):
        lines += [f"## {heading}", ""]
        rows = [i for i in items if i[0] == "row"]
        if rows:
            lines += ["| Item | Value |", "|---|---|"] + [f"| {l} | {v} |" for _, l, v in rows] + [""]
        for i in items:
            if i[0] == "p":
                lines += [i[1], ""]
            elif i[0] == "li":
                lines.append(f"- {i[1]}")
        if any(i[0] == "li" for i in items):
            lines.append("")
    return "\n".join(lines)


def to_html(res: PlanRunResult) -> str:
    e = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Personal Financial Plan</title>",
        "<style>body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:820px;margin:32px auto;"
        "padding:0 16px;color:#0b0b0b;line-height:1.5}h1{margin-bottom:0}h2{margin-top:28px;border-bottom:1px solid "
        "#e1e0d9;padding-bottom:4px}table{border-collapse:collapse;width:100%}td{padding:6px 8px;border-bottom:1px "
        "solid #eeede8}td:last-child{text-align:right;font-variant-numeric:tabular-nums}.muted{color:#52514e}"
        "@media print{body{margin:0}}</style></head><body>",
        "<h1>Personal Financial Plan</h1>",
        f"<p class='muted'>Generated {datetime.now():%d %b %Y, %H:%M} by the AI Personal Financial Advisor "
        "(academic prototype). Use your browser's Print to save as PDF.</p>",
    ]
    for heading, items in build_sections(res):
        parts.append(f"<h2>{e(heading)}</h2>")
        rows = [i for i in items if i[0] == "row"]
        if rows:
            parts.append("<table>" + "".join(f"<tr><td>{e(l)}</td><td>{e(str(v))}</td></tr>" for _, l, v in rows)
                         + "</table>")
        lis = [i for i in items if i[0] == "li"]
        if lis:
            parts.append("<ul>" + "".join(f"<li>{e(i[1])}</li>" for i in lis) + "</ul>")
        parts += [f"<p>{e(i[1])}</p>" for i in items if i[0] == "p"]
    parts.append("</body></html>")
    return "".join(parts)
