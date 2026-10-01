"""
Evaluation runner. Produces REAL numbers from the current system; nothing is
hard coded.

    python -m evaluation.evaluate --offline          # no API key needed
    python -m evaluation.evaluate                    # full run with the LLM
    python -m evaluation.evaluate --users 8          # fewer plans (cheaper)

Offline mode runs: calculation accuracy, risk classification, retrieval hit
rate (keyword or FAISS), safety detection (rules), and plan structure checks.
Full mode adds: goal parsing, tool selection, routing, RAG answers,
groundedness / hallucination (LLM judge), risk explanation consistency,
safety with the LLM judge, and end to end plan quality.

Output folder: evaluation/results/run_<timestamp>/
    summary.md        metrics table + configuration
    metrics.json      same, machine readable
    details.csv       one row per case (for the appendix of your report)
    manual_review.csv judged claims with an empty column for a human check
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import config
from agents.calculation_agent import answer_calculation_question
from agents.llm import LLMCallError, LLMClient, get_llm
from agents.profile_agent import parse_goal_text
from agents.rag_agent import format_context, run_rag_agent
from agents.risk_agent import run_risk_agent
from agents.safety_agent import check_certainty, check_text, make_plan_safety_fn, narrative_text
from evaluation.metrics import (
    Metric, groundedness, hit_at_k, score_calculation, score_goal_parse, score_safety_case, score_tool_selection,
    summary_table,
)
from graph.plan_graph import run_plan
from graph.qa_graph import ask
from prompts.agent_prompts import GROUNDEDNESS_JUDGE_SYSTEM, GROUNDEDNESS_JUDGE_USER
from rag.retriever import get_retriever
from schemas.agent_outputs import GroundednessLLM
from schemas.risk_questionnaire import parse_answers, score_answers
from tools.goal_gap import GoalPlanInputs, plan_goal
from utils.validation import split_user_row

ROOT = config.PROJECT_ROOT
DETAILS: list[dict] = []
REVIEW: list[dict] = []


def load(name: str) -> list[dict]:
    with (ROOT / "data" / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def detail(section, case_id, inp, expected, got, passed, notes=""):
    DETAILS.append({"section": section, "case_id": case_id, "input": _short(inp), "expected": _short(expected),
                    "got": _short(got), "passed": passed, "notes": notes})


def _short(x, n=300):
    s = x if isinstance(x, str) else json.dumps(x, default=str, ensure_ascii=False)
    return s[:n]


def cases_of(cases, category):
    return [(c["case_id"], json.loads(c["input_json"]), json.loads(c["expected_json"])) for c in cases
            if c["category"] == category]


def not_run(name, method, reason="needs OPENAI_API_KEY"):
    return Metric(name, method=method, status="not_run", notes=[reason])


def _plan_context(res) -> dict:
    from agents.calculation_agent import plan_context_for_qa
    ctx = plan_context_for_qa(res.profile.profile, res.profile.goal, res.calculation)
    ctx.update(effective_category=res.portfolio.allocation.effective_category.value,
               monthly_surplus=res.profile.monthly_surplus,
               debt_to_income_pct=res.calculation.health["debt_to_income"]["value"],
               dependents=res.profile.profile.dependents)
    return ctx


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def eval_calculation(cases) -> Metric:
    m = Metric("Calculation accuracy", method="Tool output vs independent simulation ground truth, tolerance Rs 1")
    for cid, inp, exp in cases_of(cases, "calculation"):
        ok, errors = score_calculation(plan_goal(GoalPlanInputs(**inp)).to_dict(), exp)
        m.add(ok)
        detail("calculation", cid, inp, exp, errors or "match", ok)
    return m


def eval_risk_rules(cases, users) -> Metric:
    m = Metric("Risk classification (rules)", method="6 boundary cases + 24 users, questionnaire score vs label")
    for cid, inp, exp in cases_of(cases, "risk_classification"):
        r = score_answers(parse_answers(inp["answers"]))
        ok = (r.score, r.category.value) == (exp["score"], exp["category"])
        m.add(ok)
        detail("risk_rules", cid, inp, exp, [r.score, r.category.value], ok)
    for u in users:
        r = score_answers(parse_answers(u["risk_answers"]))
        m.add(r.category.value == u["risk_profile"])
    m.notes.append("Deterministic by design; 100% expected. Consistency of the AI explanation is measured separately.")
    return m


def eval_risk_llm_consistency(users, llm, repeats: int) -> Metric:
    m = Metric("Risk explanation consistency (LLM)",
               method=f"Risk agent run {repeats}x on 5 users; pass if the category never changes and every "
                      "explanation names the correct category")
    for u in users[:5]:
        p, g = split_user_row(u)
        answers = parse_answers(u["risk_answers"])
        outs = [run_risk_agent(answers, llm) for _ in range(repeats)]
        cats = {o.risk.category.value for o in outs}
        mentions = [bool(o.risk.reasoning) and u["risk_profile"].lower() in o.risk.reasoning.lower() for o in outs]
        ok = len(cats) == 1 and all(mentions)
        m.add(ok)
        detail("risk_llm", u["user_id"], answers, u["risk_profile"], {"categories": sorted(cats), "mentions": mentions}, ok)
    return m


def eval_goal_parsing(cases, users, llm) -> list[Metric]:
    full = Metric("Goal parsing: all stated fields correct", method="7 labelled free text goals")
    typ = Metric("Goal classification accuracy (goal type)", method="7 labelled goals + 24 user goal descriptions")
    for cid, inp, exp in cases_of(cases, "goal_parsing"):
        try:
            got = parse_goal_text(inp["text"], llm).model_dump(mode="json")
        except LLMCallError as e:
            got = {"error": str(e)}
        ok, fields = score_goal_parse(got, exp)
        full.add(ok)
        typ.add(fields.get("goal_type", False))
        detail("goal_parsing", cid, inp["text"], exp, got, ok, json.dumps(fields))
    for u in users:
        try:
            got = parse_goal_text(u["goal_description"], llm, int(u["age"])).model_dump(mode="json")
            ok = got["goal_type"] == u["goal_type"]
        except LLMCallError as e:
            got, ok = {"error": str(e)}, False
        typ.add(ok)
        detail("goal_type_users", u["user_id"], u["goal_description"], u["goal_type"], got.get("goal_type", got), ok)
    return [full, typ]


def eval_tool_selection(cases, llm, context) -> list[Metric]:
    any_m = Metric("Tool selection accuracy", method="8 labelled questions; expected tool among the tools called")
    first = Metric("Tool selection (first call correct)", method="Expected tool is the first tool called")
    for cid, inp, exp in cases_of(cases, "tool_selection"):
        out = answer_calculation_question(inp["question"], context, llm)
        called = [t.tool for t in out.tool_calls]
        ok_any, ok_first = score_tool_selection(called, exp["tool"])
        any_m.add(ok_any)
        first.add(ok_first)
        detail("tool_selection", cid, inp["question"], exp["tool"], called or out.llm_error, ok_any)
    return [any_m, first]


def eval_routing(cases, llm, retriever, plan_res) -> Metric:
    m = Metric("Q&A routing accuracy", method="8 labelled questions through the LangGraph router")
    for cid, inp, exp in cases_of(cases, "routing"):
        r = ask(inp["question"], llm, retriever, plan_res.plan, _plan_context(plan_res))
        ok = r.route == exp["route"]
        m.add(ok)
        detail("routing", cid, inp["question"], exp["route"], r.route, ok)
    return m


def eval_retrieval(cases, retriever) -> list[Metric]:
    k = config.RAG_TOP_K
    h1 = Metric("Retrieval hit@1", method=f"{retriever.method}; expected source ranked first")
    hk = Metric(f"Retrieval hit@{k}", method=f"{retriever.method}; expected source in top {k}")
    for cid, inp, exp in cases_of(cases, "rag_retrieval"):
        hits = retriever.retrieve(inp["question"], k=k)
        sources = [h.source_file for h in hits]
        h1.add(hit_at_k(sources, exp["expected_source"], 1))
        hk.add(hit_at_k(sources, exp["expected_source"], k))
        detail("retrieval", cid, inp["question"], exp["expected_source"],
               [f"{h.source_file}:{h.score}" for h in hits], hit_at_k(sources, exp["expected_source"], k))
    return [h1, hk]


def judge_claims(judge: LLMClient, text: str, facts: dict, chunks, label: str) -> tuple[int, int]:
    res = judge.structured(GroundednessLLM, GROUNDEDNESS_JUDGE_SYSTEM, GROUNDEDNESS_JUDGE_USER.format(
        facts=json.dumps(facts, default=str, ensure_ascii=False)[:6000], context=format_context(chunks)[:6000],
        text=text))
    claims = [c.model_dump() for c in res.claims]
    for c in claims:
        REVIEW.append({"source": label, "claim": c["claim"], "judge_supported": c["supported"], "human_supported": ""})
    return groundedness(claims)


def eval_rag_answers(cases, retriever, llm, judge, counters) -> list[Metric]:
    cite = Metric("RAG citation accuracy", method="Answer is answerable and cites the expected knowledge file")
    ground = Metric("RAG groundedness", method="LLM judge: claims supported by retrieved chunks / all claims")
    invented = 0
    for cid, inp, exp in cases_of(cases, "rag_retrieval"):
        out = run_rag_agent(inp["question"], retriever, llm)
        ok = out.answerable and exp["expected_source"] in out.sources_used
        cite.add(ok)
        invented += len(out.invented_sources_removed)
        detail("rag_answer", cid, inp["question"], exp["expected_source"],
               {"cited": out.sources_used, "answer": out.answer}, ok)
        if out.answer and out.answerable:
            try:
                s, t = judge_claims(judge, out.answer, {}, out.retrieved, f"rag:{cid}")
                ground.passed += s
                ground.total += t
                counters["supported"] += s
                counters["claims"] += t
            except LLMCallError as e:
                ground.notes.append(f"{cid} judge failed: {e}")
    cite.notes.append(f"Invented sources caught and removed: {invented}")
    return [cite, ground]


def eval_safety_labelled(cases, llm) -> list[Metric]:
    out = []
    for use_llm in ([False, True] if llm else [False]):
        name = "Safety detection (rules + LLM judge)" if use_llm else "Safety detection (rules only)"
        m = Metric(name, method="8 labelled texts: planted violations detected, clean text not blocked")
        extras = 0
        for cid, inp, exp in cases_of(cases, "safety"):
            r = check_text(inp["text"], inp.get("context"), llm=llm if use_llm else None, use_llm=use_llm)
            got = {f.category.value for f in r.flags}
            ok, extra = score_safety_case(got, set(exp["flags"]), not r.passed)
            extras += len(extra)
            m.add(ok)
            detail("safety_llm" if use_llm else "safety_rules", cid, inp["text"], exp["flags"], sorted(got), ok)
        m.notes.append(f"Extra categories flagged beyond labels: {extras}")
        if not use_llm:
            m.notes.append("Rules were developed while viewing these 8 cases; use an unseen set for an unbiased figure")
        out.append(m)
    return out


def eval_safety_unseen(llm) -> Metric | None:
    """Optional: data/safety_unseen.csv written by someone who has NOT seen the rules.
    Columns: text, expected_flags (comma separated categories, empty for a clean text)."""
    path = ROOT / "data" / "safety_unseen.csv"
    if not path.exists():
        return None
    m = Metric("Safety detection on unseen texts", method="Texts written without seeing the rules"
               + (" (rules + LLM judge)" if llm else " (rules only)"))
    with path.open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            expected = {x.strip() for x in (row.get("expected_flags") or "").split(",") if x.strip()}
            r = check_text(row["text"], llm=llm)
            got = {fl.category.value for fl in r.flags}
            ok, _ = score_safety_case(got, expected, not r.passed)
            m.add(ok)
            detail("safety_unseen", f"S{i}", row["text"], sorted(expected), sorted(got), ok)
    return m


def eval_plans(users, llm, retriever, judge, counters) -> list[Metric]:
    ok_m = Metric("Plans generated without error", method=f"Full LangGraph run for {len(users)} users")
    disc = Metric("Disclaimer and assumptions present", method="Every plan shows disclaimer, return, inflation, "
                                                                "horizon and contribution frequency")
    clean = Metric("Final plan text free of guaranteed-return language", method="Rule re-check of the final "
                                                                                "narrative (withheld text counts as clean)")
    safe = Metric("Plans passing safety", method="Safety status passed (first try or after revision)")
    ground = Metric("Plan narrative groundedness", method="LLM judge: claims supported by plan facts + context")
    statuses: dict[str, int] = {}
    revisions, latencies = 0, []
    for u in users:
        p, g = split_user_row(u)
        t0 = time.perf_counter()
        res = run_plan(p, g, parse_answers(u["risk_answers"]), llm, retriever, safety_fn=make_plan_safety_fn(llm))
        latencies.append(time.perf_counter() - t0)
        ok_m.add(res.ok)
        if not res.ok:
            detail("plan", u["user_id"], u["goal_type"], "ok", res.errors, False)
            continue
        plan = res.plan
        asm = plan.assumptions
        disc.add(bool(plan.disclaimer) and all(asm.get(k) not in (None, "") for k in
                                               ("expected_annual_return_pct", "annual_inflation_pct", "horizon_years",
                                                "contribution_frequency")))
        text = narrative_text(plan)
        clean.add(not check_certainty(text))
        statuses[res.safety_status] = statuses.get(res.safety_status, 0) + 1
        revisions += res.revision_count
        if llm is not None:
            safe.add(res.safety_status == "passed")
        if text and judge is not None:
            facts = {k: getattr(plan, k) for k in ("snapshot", "goal", "risk", "goal_calculation", "allocation",
                                                   "assumptions")}
            try:
                s, t = judge_claims(judge, text, facts, res.context_chunks, f"plan:{u['user_id']}")
                ground.passed += s
                ground.total += t
                counters["supported"] += s
                counters["claims"] += t
            except LLMCallError as e:
                ground.notes.append(f"{u['user_id']} judge failed: {e}")
        detail("plan", u["user_id"], u["goal_type"], "passed", res.safety_status, res.safety_status.startswith("passed"),
               f"revisions={res.revision_count}")
    safe.notes.append(f"Status counts: {statuses}; total revisions: {revisions}")
    ok_m.notes.append(f"Average time per plan: {sum(latencies) / max(len(latencies), 1):.1f}s")
    if llm is None:
        safe.status = ground.status = clean.status = "not_run"
        clean.notes.insert(0, "needs OPENAI_API_KEY (offline plans contain no AI text)")
        safe.notes.insert(0, "needs OPENAI_API_KEY (without it there is no AI text to check)")
        ground.notes.insert(0, "needs OPENAI_API_KEY")
    return [ok_m, disc, clean, safe, ground]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the AI Personal Financial Advisor")
    ap.add_argument("--offline", action="store_true", help="Run only the parts that need no API key")
    ap.add_argument("--users", type=int, default=24, help="Number of sample users for end to end plans")
    ap.add_argument("--repeats", type=int, default=3, help="Repeats for the risk consistency test")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    args = ap.parse_args(argv)

    DETAILS.clear()
    REVIEW.clear()
    cases, users = load("test_cases.csv"), load("sample_users.csv")
    llm = None if args.offline else get_llm()
    judge = None
    if llm is not None:
        judge = LLMClient(model_name=config.JUDGE_MODEL_NAME)
    elif not args.offline:
        print("No OPENAI_API_KEY found: running offline parts only.")
    retriever = get_retriever()
    counters = {"supported": 0, "claims": 0}

    print("Running evaluation" + (" (offline)" if llm is None else f" with {config.MODEL_NAME}") + "...")
    metrics: list[Metric] = [eval_calculation(cases), eval_risk_rules(cases, users)]
    metrics += eval_retrieval(cases, retriever)
    metrics += eval_safety_labelled(cases, llm)
    unseen = eval_safety_unseen(llm)
    if unseen:
        metrics.append(unseen)

    if llm is not None:
        metrics.append(eval_risk_llm_consistency(users, llm, args.repeats))
        metrics += eval_goal_parsing(cases, users, llm)
        u02 = next(u for u in users if u["user_id"] == "U02")
        p, g = split_user_row(u02)
        base = run_plan(p, g, parse_answers(u02["risk_answers"]), llm, retriever)
        metrics += eval_tool_selection(cases, llm, _plan_context(base))
        metrics.append(eval_routing(cases, llm, retriever, base))
        metrics += eval_rag_answers(cases, retriever, llm, judge, counters)
    else:
        metrics += [
            not_run("Risk explanation consistency (LLM)", "Repeated risk agent runs"),
            not_run("Goal classification accuracy (goal type)", "31 labelled goal texts"),
            not_run("Tool selection accuracy", "8 labelled questions"),
            not_run("Q&A routing accuracy", "8 labelled questions"),
            not_run("RAG citation accuracy", "7 labelled questions"),
            not_run("RAG groundedness", "LLM judge"),
        ]
    metrics += eval_plans(users[: args.users], llm, retriever, judge, counters)

    halluc = Metric("Hallucination rate (unsupported claims)", method="Unsupported / all judged claims "
                                                                       "(RAG answers + plan narratives)")
    if counters["claims"]:
        halluc.passed = counters["claims"] - counters["supported"]
        halluc.total = counters["claims"]
        halluc.notes.append("LOWER is better. Judge is an LLM; verify with manual_review.csv")
    else:
        halluc.status = "not_run"
        halluc.notes.append("needs OPENAI_API_KEY")
    metrics.append(halluc)

    # ---------------------------------------------------------------- write
    out = Path(args.out) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = {"mode": "offline" if llm is None else "full", "model": config.MODEL_NAME if llm else None,
               "judge_model": config.JUDGE_MODEL_NAME if llm else None, "retriever": retriever.method,
               "rag_min_score": config.RAG_MIN_SCORE, "users": min(args.users, len(users)),
               "risk_repeats": args.repeats, "timestamp": datetime.now().isoformat(timespec="seconds")}
    table = summary_table(metrics)
    (out / "summary.md").write_text(
        f"# Evaluation results\n\n**Configuration:** `{json.dumps(run_cfg)}`\n\n{table}\n\n"
        "Rates are passed / total. For the hallucination rate, the count shown is UNSUPPORTED claims "
        "(lower is better).\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"config": run_cfg, "metrics": [m.as_row() for m in metrics]},
                                                 indent=2), encoding="utf-8")
    with (out / "details.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "case_id", "input", "expected", "got", "passed", "notes"])
        w.writeheader()
        w.writerows(DETAILS)
    if REVIEW:
        with (out / "manual_review.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "claim", "judge_supported", "human_supported"])
            w.writeheader()
            w.writerows(REVIEW)
    print("\n" + table + f"\n\nSaved to {out}")
    return out


if __name__ == "__main__":
    main()
