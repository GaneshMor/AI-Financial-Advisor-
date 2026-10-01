"""
LIVE check of the LLM parts. Needs OPENAI_API_KEY in .env. Costs a few rupees.

    python -m scripts.smoke_test_llm

1. Goal parsing on the 7 labelled goal_parsing cases
2. Tool calling on the 8 labelled tool_selection cases
3. Q&A routing through the LangGraph Q&A graph
4. RAG answers on the 7 labelled knowledge questions
5. Full plan pipeline for user U02 (narrative written by the LLM)

This is a quick smoke test, not the evaluation. Phase 9 produces the real metrics.
"""

import csv
import json
from pathlib import Path

from agents.calculation_agent import answer_calculation_question, plan_context_for_qa
from agents.llm import get_llm
from agents.pipeline import run_plan_pipeline
from agents.profile_agent import parse_goal_text
from graph.qa_graph import ask
from agents.rag_agent import run_rag_agent
from rag.retriever import get_retriever
from agents.safety_agent import make_plan_safety_fn, make_qa_safety_fn
from agents.llm import LLMCallError
from schemas.risk_questionnaire import parse_answers
from utils.validation import split_user_row, validate_inputs

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    llm = get_llm()
    if llm is None:
        raise SystemExit("OPENAI_API_KEY is not set in .env. Add it and run again.")
    cases = load("test_cases.csv")
    retriever = get_retriever()
    print(f"Retriever: {retriever.method}" + (f"  (note: {retriever.warning})" if retriever.warning else "") + "\n")

    print("=== 1. GOAL PARSING ===")
    ok = total = 0
    for c in (c for c in cases if c["category"] == "goal_parsing"):
        text = json.loads(c["input_json"])["text"]
        exp = json.loads(c["expected_json"])
        try:
            got = parse_goal_text(text, llm).model_dump(mode="json")
        except LLMCallError as e:
            print(f"{c['case_id']} ERROR {e}")
            total += 1
            continue
        fields = [k for k in ("goal_type", "goal_amount", "horizon_years", "amount_is_present_value")
                  if exp.get(k) is not None]
        match = all(got.get(k) == exp[k] for k in fields)
        ok += match
        total += 1
        print(f"{c['case_id']} {'PASS' if match else 'FAIL'}  {text!r}\n      got={ {k: got.get(k) for k in fields} }")
    print(f"Goal parsing: {ok}/{total}\n")

    print("=== 2. TOOL SELECTION ===")
    row = next(u for u in load("sample_users.csv") if u["user_id"] == "U02")
    rep = validate_inputs(*split_user_row(row))
    res = run_plan_pipeline(rep.profile, rep.goal, parse_answers(row["risk_answers"]), llm, retriever,
                            safety_fn=make_plan_safety_fn(llm))
    context = plan_context_for_qa(rep.profile, rep.goal, res.calculation)
    ok = total = 0
    for c in (c for c in cases if c["category"] == "tool_selection"):
        q = json.loads(c["input_json"])["question"]
        exp = json.loads(c["expected_json"])["tool"]
        out = answer_calculation_question(q, context, llm)
        called = [t.tool for t in out.tool_calls]
        match = exp in called
        ok += match
        total += 1
        print(f"{c['case_id']} {'PASS' if match else 'FAIL'}  {q!r} -> {called or out.llm_error}")
    print(f"Tool selection: {ok}/{total}\n")

    print("=== 3. Q&A ROUTER (LangGraph) ===")
    route_cases = [
        ("Can I achieve my goal with my current SIP?", "calculation"),
        ("What happens if inflation increases to 8%?", "calculation"),
        ("What is diversification?", "knowledge"),
        ("Why is my allocation moderate?", "knowledge"),
        ("Which stock should I buy to double my money?", "out_of_scope"),
        ("How do I file my income tax return?", "out_of_scope"),
    ]
    ok = 0
    for q, expected in route_cases:
        r = ask(q, llm, retriever=retriever, plan=res.plan, plan_context=context,
                safety_fn=make_qa_safety_fn(llm))
        ok += r.route == expected
        print(f"{'PASS' if r.route == expected else 'FAIL'}  {q!r} -> {r.route} | trace={r.trace}")
        if r.safety_flags:
            print("      safety flags:", r.safety_flags)
        if r.answer:
            print("      answer:", r.answer[:160].replace("\n", " "))
    print(f"Routing: {ok}/{len(route_cases)}\n")

    print("=== 4. RAG ANSWERS (labelled rag_retrieval cases) ===")
    ok = total = 0
    for c in (c for c in cases if c["category"] == "rag_retrieval"):
        q = json.loads(c["input_json"])["question"]
        exp = json.loads(c["expected_json"])["expected_source"]
        out = run_rag_agent(q, retriever, llm)
        match = out.answerable and exp in out.sources_used
        ok += match
        total += 1
        print(f"{c['case_id']} {'PASS' if match else 'FAIL'}  {q!r} -> cited {out.sources_used}"
              + (f" | removed invented {out.invented_sources_removed}" if out.invented_sources_removed else ""))
        print("      answer:", (out.answer or out.llm_error or "")[:200].replace("\n", " "))
    print(f"RAG: {ok}/{total}\n")

    print("=== 5. FULL PLAN (U02) ===")
    print("Sources cited:", res.plan.sources)
    print("Safety status:", res.safety_status, "| trace:", " -> ".join(res.trace))
    plan = res.plan
    print("Profile LLM:", res.profile.llm_used, res.profile.llm_error or "")
    print("Risk LLM:", res.risk.llm_used, res.risk.llm_error or "")
    print("Portfolio LLM:", res.portfolio.llm_used, res.portfolio.llm_error or "")
    if plan.narrative:
        print(json.dumps(plan.narrative.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("Narrative failed:", plan.llm_error)


if __name__ == "__main__":
    main()
