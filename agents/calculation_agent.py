"""
AGENT 3: Financial Calculation Agent.

Two modes:

1. Plan mode  (run_calculation_agent)
   Fixed sequence of Python tool calls, no LLM. Every call is recorded as a
   ToolCallRecord so the UI and log show exactly which tool produced which
   number.

2. Q&A mode   (answer_calculation_question)
   REAL tool calling: the LLM reads the question, CHOOSES a tool and fills the
   arguments; Python executes it; the LLM then phrases the answer using the
   tool result. This is where "Tool Selection Accuracy" is measured.
"""

import json
from dataclasses import asdict

import config
from agents.llm import LLMCallError, LLMClient, message_text
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from prompts.agent_prompts import QA_TOOL_SYSTEM, QA_TOOL_USER
from schemas.agent_outputs import CalculationAgentOutput, QAToolAnswer, ToolCallRecord
from schemas.models import AllocationResult, Assumptions, GoalInput, UserProfile
from tools.financial_health import financial_health
from tools.goal_gap import GoalPlanInputs, plan_goal
from tools.lc_tools import tools_by_name
from tools.scenario import standard_scenarios
from utils.logging import SessionLog, Timer

AGENT = "calculation_agent"


def _record(log, records, tool, inputs, fn):
    """Run one tool, record it, and return its result (or None on error)."""
    with Timer() as t:
        try:
            result = fn()
            rec = ToolCallRecord(tool=tool, inputs=inputs, output=_to_jsonable(result))
        except (ValueError, TypeError) as e:
            result = None
            rec = ToolCallRecord(tool=tool, inputs=inputs, success=False, error=str(e))
    records.append(rec)
    if log is not None:
        log.record(AGENT, "tool_call", tool=tool, inputs=inputs,
                   output_summary="ok" if rec.success else "error",
                   status="ok" if rec.success else "error", error=rec.error, duration_ms=t.ms)
    return result


def _to_jsonable(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_to_jsonable(o) for o in obj]
    return obj


def run_calculation_agent(
    profile: UserProfile,
    goal: GoalInput,
    allocation: AllocationResult,
    log: SessionLog | None = None,
    expected_return_pct: float | None = None,
    inflation_pct: float | None = None,
) -> CalculationAgentOutput:
    """Deterministic plan calculations. Raises ValueError only if inputs are invalid."""
    assumptions = Assumptions(
        expected_return_pct=allocation.weighted_return_pct if expected_return_pct is None else expected_return_pct,
        inflation_pct=config.DEFAULT_INFLATION_PCT if inflation_pct is None else inflation_pct,
    )
    records: list[ToolCallRecord] = []

    health_in = dict(
        monthly_income=profile.monthly_income, monthly_expenses=profile.monthly_expenses,
        monthly_debt=profile.monthly_debt, monthly_investment=profile.monthly_investment,
        emergency_fund=profile.emergency_fund,
    )
    health = _record(log, records, "financial_health", health_in, lambda: financial_health(**health_in))

    plan_inputs = GoalPlanInputs(
        goal_amount=goal.goal_amount,
        amount_is_present_value=goal.amount_is_present_value,
        horizon_years=goal.horizon_years,
        annual_inflation_pct=assumptions.inflation_pct,
        annual_return_pct=assumptions.expected_return_pct,
        current_goal_savings=goal.current_goal_savings,
        current_monthly_sip=profile.monthly_investment,
        free_investable_surplus=health.free_investable_surplus if health else 0.0,
    )
    plan_in = asdict(plan_inputs)
    plan = _record(log, records, "plan_goal", plan_in, lambda: plan_goal(plan_inputs))
    scenarios = _record(log, records, "standard_scenarios", plan_in, lambda: standard_scenarios(plan_inputs))

    if plan is None:
        failed = next(r for r in records if not r.success)
        raise ValueError(f"Goal calculation failed: {failed.error}")

    return CalculationAgentOutput(
        assumptions=assumptions,
        health=health.to_dict() if health else {},
        goal_plan=plan.to_dict(),
        scenarios=scenarios or [],
        tool_calls=records,
    )


def plan_context_for_qa(profile: UserProfile, goal: GoalInput, calc: CalculationAgentOutput) -> dict:
    """The facts the Q&A agent may use to fill tool arguments."""
    gp = calc.goal_plan
    return {
        "goal_type": goal.goal_type.value,
        "goal_amount": goal.goal_amount,
        "amount_is_present_value": goal.amount_is_present_value,
        "horizon_years": goal.horizon_years,
        "current_goal_savings": goal.current_goal_savings,
        "current_monthly_sip": profile.monthly_investment,
        "free_investable_surplus": round(calc.health.get("free_investable_surplus", 0.0), 2),
        "monthly_income": profile.monthly_income,
        "monthly_expenses": profile.monthly_expenses,
        "monthly_debt": profile.monthly_debt,
        "emergency_fund": profile.emergency_fund,
        "assumed_annual_return_pct": calc.assumptions.expected_return_pct,
        "assumed_annual_inflation_pct": calc.assumptions.inflation_pct,
        "current_required_monthly_sip": round(gp["required_monthly_sip"], 2),
        "current_status": gp["gap"]["status"],
    }


def answer_calculation_question(
    question: str,
    plan_context: dict | None,
    llm: LLMClient | None,
    log: SessionLog | None = None,
) -> QAToolAnswer:
    """Tool calling loop: LLM picks tools, Python runs them, LLM writes the answer."""
    out = QAToolAnswer(question=question)
    if llm is None:
        out.llm_error = "LLM not configured. Use the calculators on the Goal Analysis page."
        return out

    tools = tools_by_name()
    try:
        model = llm.bind_tools(list(tools.values()))
        messages = [
            SystemMessage(content=QA_TOOL_SYSTEM),
            HumanMessage(content=QA_TOOL_USER.format(
                context=json.dumps(plan_context or {}, indent=2), question=question)),
        ]
        ai = None
        for _ in range(config.QA_MAX_TOOL_ROUNDS):
            ai = model.invoke(messages)
            messages.append(ai)
            if not getattr(ai, "tool_calls", None):
                break
            for call in ai.tool_calls:
                name, args = call["name"], call.get("args", {})
                tool = tools.get(name)
                if tool is None:
                    result = {"error": f"Unknown tool '{name}'."}
                else:
                    result = tool.invoke(args)
                ok = not (isinstance(result, dict) and "error" in result)
                out.tool_calls.append(ToolCallRecord(tool=name, inputs=args, output=result,
                                                     success=ok, error=None if ok else result["error"]))
                if log is not None:
                    log.record(AGENT, "qa_tool_call", tool=name, inputs=args,
                               output_summary="ok" if ok else "error", status="ok" if ok else "error")
                messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
        else:
            # Tool round limit hit: ask once more for a final text answer, tools disabled
            ai = llm.bind_tools(list(tools.values()), tool_choice="none").invoke(messages)

        out.answer = message_text(ai).strip() or None
        if out.answer is None:
            out.llm_error = "The model returned no answer text."
    except LLMCallError as e:
        out.llm_error = str(e)
    except Exception as e:  # API errors raised by the bound model
        out.llm_error = f"{type(e).__name__}: {e}"
    return out
