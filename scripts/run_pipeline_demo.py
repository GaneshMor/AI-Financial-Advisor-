"""
Run every plan agent for one sample user and print the result.

    python -m scripts.run_pipeline_demo            # user U02
    python -m scripts.run_pipeline_demo U17        # any user id

With OPENAI_API_KEY in .env the LLM parts run for real.
Without it, the numbers still run and the LLM parts show a clear fallback.
The knowledge base is added in Phase 6, so RAG context is empty for now.
"""

import csv
import json
import sys
from pathlib import Path

from agents.llm import get_llm
from agents.pipeline import run_plan_pipeline
from rag.retriever import get_retriever
from agents.safety_agent import make_plan_safety_fn
from schemas.risk_questionnaire import parse_answers
from utils.formatting import format_inr
from utils.logging import SessionLog
from utils.validation import split_user_row, validate_inputs

ROOT = Path(__file__).resolve().parent.parent


def load_user(user_id: str) -> dict:
    with (ROOT / "data" / "sample_users.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["user_id"] == user_id:
                return row
    raise SystemExit(f"User {user_id} not found in data/sample_users.csv")


def main(user_id: str = "U02") -> None:
    row = load_user(user_id)
    report = validate_inputs(*split_user_row(row))
    if not report.ok:
        raise SystemExit("Invalid input: " + "; ".join(report.errors))

    llm = get_llm()
    print(f"LLM: {'connected' if llm else 'NOT configured (fallback mode)'}\n")
    retriever = get_retriever()
    print(f"Retriever: {retriever.method if retriever else 'none'}")
    if retriever and retriever.warning:
        print(f"Note: {retriever.warning}")
    log = SessionLog()
    res = run_plan_pipeline(report.profile, report.goal, parse_answers(row["risk_answers"]), llm, retriever, log,
                            safety_fn=make_plan_safety_fn(llm))
    plan = res.plan
    print("LangGraph trace:", " -> ".join(res.trace))
    print("Safety status:", res.safety_status)

    gc, al = plan.goal_calculation, plan.allocation
    print(f"=== PLAN FOR {user_id}: {plan.goal['goal_type']} in {plan.goal['horizon_years']} years ===")
    print("Risk:", plan.risk["score"], plan.risk["category"], "| effective:", al["effective_category"],
          "| guardrail:", al["guardrail_applied"])
    print(f"Allocation (illustrative): equity {al['equity_pct']:.0f}% / debt {al['debt_pct']:.0f}% / gold {al['gold_pct']:.0f}%"
          f"  -> assumed return {al['weighted_assumed_return_pct']}%")
    print("Target:", format_inr(gc["target_future_value"]), "| Required SIP:", format_inr(gc["required_monthly_sip"]),
          "| Current SIP:", format_inr(gc["current_monthly_sip"]), "| Status:", gc["status"])
    print("Professional advice recommended:", plan.professional_advice_recommended, plan.escalation_reasons)
    print("Warnings:", plan.warnings)
    print("Risk factors:", plan.risk["key_risk_factors"], f"(source: {plan.risk['factors_source']})")

    if res.profile.parsed_goal:
        print("\nParsed goal (LLM):", res.profile.parsed_goal.model_dump(mode="json"))
        print("Discrepancies:", res.profile.discrepancies)
    if plan.narrative:
        print("\n--- AI narrative ---")
        print(json.dumps(plan.narrative.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("\nAI narrative unavailable:", plan.llm_error)

    print("\n--- Log (field names only, no values) ---")
    for e in log.to_records():
        print(f"{e['agent']:<18} {e['action']:<22} tool={e['tool']!s:<20} {e['status']:<8} {e['output_summary']}")
    print("\nKnowledge sources given to the advisor:", sorted({c.source_file for c in res.context_chunks}))
    print("Sources cited in narrative:", plan.sources)
    print("\n" + plan.disclaimer)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "U02")
