"""
Session state and shared services for the Streamlit app.

Everything the user enters lives in st.session_state as plain dicts:
    profile_data, goal_data, risk_answers, assumptions
The plan is generated on demand (button) and cached with a hash of the inputs,
so the app knows when the plan is out of date.
"""

import csv
import hashlib
import json

import streamlit as st

import config
from agents.calculation_agent import plan_context_for_qa
from agents.llm import get_llm
from agents.safety_agent import make_plan_safety_fn, make_qa_safety_fn
from graph.plan_graph import PlanRunResult, run_plan
from rag.retriever import get_retriever
from schemas.risk_questionnaire import parse_answers
from utils.logging import SessionLog
from utils.validation import split_user_row

SAMPLE_USERS_PATH = config.PROJECT_ROOT / "data" / "sample_users.csv"

DEFAULTS = {
    "profile_data": None,
    "goal_data": None,
    "risk_answers": None,
    "assumptions": {"inflation_pct": config.DEFAULT_INFLATION_PCT, "override_return": False,
                    "expected_return_pct": 10.0},
    "plan_result": None,
    "plan_hash": None,
    "qa_history": [],
}


def init_state() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = json.loads(json.dumps(v)) if isinstance(v, (dict, list)) else v
    if "log" not in st.session_state:
        st.session_state.log = SessionLog()


def reset_all() -> None:
    for k in list(DEFAULTS) + ["log"]:
        st.session_state.pop(k, None)
    init_state()


# ---------------------------------------------------------------------------
# Shared services (created once per server process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def llm_client():
    return get_llm()


@st.cache_resource(show_spinner=False)
def retriever():
    return get_retriever()


def service_status() -> dict:
    llm = llm_client()
    r = retriever()
    return {
        "llm": llm is not None,
        "model": config.MODEL_NAME if llm else None,
        "retriever": r.method if r else "none",
        "retriever_warning": getattr(r, "warning", None) if r else "Knowledge base not found.",
    }


# ---------------------------------------------------------------------------
# Sample users (for demos)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def sample_users() -> list[dict]:
    with SAMPLE_USERS_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sample_user(user_id: str) -> None:
    row = next(u for u in sample_users() if u["user_id"] == user_id)
    profile, goal = split_user_row(row)
    ints = {"age", "dependents"}
    st.session_state.profile_data = {k: (int(v) if k in ints else float(v)) for k, v in profile.items()}
    goal["goal_amount"] = float(goal["goal_amount"])
    goal["current_goal_savings"] = float(goal["current_goal_savings"])
    goal["horizon_years"] = int(goal["horizon_years"])
    st.session_state.goal_data = goal
    st.session_state.risk_answers = parse_answers(row["risk_answers"])
    st.session_state.plan_result = None
    st.session_state.qa_history = []


# ---------------------------------------------------------------------------
# Progress and plan generation
# ---------------------------------------------------------------------------
def steps_done() -> dict[str, bool]:
    s = st.session_state
    return {"profile": s.profile_data is not None, "goal": s.goal_data is not None,
            "risk": s.risk_answers is not None, "plan": plan_is_current()}


def inputs_complete() -> bool:
    d = steps_done()
    return d["profile"] and d["goal"] and d["risk"]


def _inputs_hash() -> str:
    s = st.session_state
    payload = json.dumps([s.profile_data, s.goal_data, s.risk_answers, s.assumptions], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def plan_is_current() -> bool:
    return st.session_state.plan_result is not None and st.session_state.plan_hash == _inputs_hash()


def generate_plan() -> PlanRunResult:
    s = st.session_state
    llm = llm_client()
    a = s.assumptions
    res = run_plan(
        s.profile_data, s.goal_data, s.risk_answers, llm, retriever(),
        safety_fn=make_plan_safety_fn(llm), log=s.log,
        expected_return_pct=a["expected_return_pct"] if a["override_return"] else None,
        inflation_pct=a["inflation_pct"],
    )
    s.plan_result = res
    s.plan_hash = _inputs_hash()
    s.qa_history = []
    return res


def plan() -> PlanRunResult | None:
    return st.session_state.plan_result


def qa_context(res: PlanRunResult) -> dict:
    """Facts the Q&A agent may use, plus the fields the safety check needs."""
    ctx = plan_context_for_qa(res.profile.profile, res.profile.goal, res.calculation)
    ctx.update({
        "effective_category": res.portfolio.allocation.effective_category.value,
        "monthly_surplus": res.profile.monthly_surplus,
        "debt_to_income_pct": res.calculation.health["debt_to_income"]["value"],
        "dependents": res.profile.profile.dependents,
    })
    return ctx


def qa_safety_fn():
    return make_qa_safety_fn(llm_client())
