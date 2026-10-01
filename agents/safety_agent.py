"""
AGENT 6: Responsible AI / Safety Agent.

Two layers:
  1. RULE CHECKS (deterministic, always run, no API needed)
     A  guaranteed_return   : "guaranteed", "definitely", "assured returns", "will give 15%"
     B  unsupported_claim   : "risk free", "no risk", "cannot lose", "always beats"
     C  risk_mismatch       : equity above the limit for the effective category / short horizon
     D  hallucination       : named funds or products, invented regulations, historical returns,
                              invented sources, numbers not found in calculations or sources
     E  missing_assumption  : a projection with no stated return/inflation assumption
     F  escalation          : stressed situation but no suggestion to consult a professional
  2. LLM JUDGE (optional): a second model call looks for unsupported claims the rules
     cannot see. If it fails, the rule results still stand (checked_by_llm = False).

Severity HIGH means the text must not be shown as is. In the plan graph a HIGH
flag triggers a rewrite; in Q&A the answer is withheld.
"""

import json
import re
from typing import Iterable

import config
from agents.llm import LLMCallError, LLMClient
from agents.rag_agent import format_context
from prompts.agent_prompts import SAFETY_JUDGE_SYSTEM, SAFETY_JUDGE_USER
from schemas.agent_outputs import FinalPlan, SafetyJudgeLLM
from schemas.models import RetrievedChunk, SafetyCategory, SafetyFlag, SafetyReport, Severity

H, M = Severity.HIGH, Severity.MEDIUM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NEGATIONS = re.compile(r"\b(not|no|never|cannot|can't|isn't|aren't|won't|nor|without)\b[^.]{0,25}$", re.I)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip(" -•\t") for p in parts if p.strip(" -•\t")]


def _negated(sentence: str, start: int) -> bool:
    """True if a negation word appears shortly before position `start`."""
    return bool(NEGATIONS.search(sentence[max(0, start - 40):start]))


def _flag(cat: SafetyCategory, sev: Severity, text: str, suggestion: str) -> SafetyFlag:
    return SafetyFlag(category=cat, severity=sev, text=text[:300], suggestion=suggestion)


# ---------------------------------------------------------------------------
# A + B: certainty language
# ---------------------------------------------------------------------------
GUARANTEE_PATTERNS = [
    r"\bguarantee(d|s)?\b", r"\bdefinitely\b", r"\bassured\s+returns?\b", r"\bcertainly\b",
    r"\bsure[\s-]?shot\b", r"\bwill\s+(?:surely|certainly|definitely)\b",
    r"\bwill\s+(?:give|earn|generate|deliver|return|fetch)\b[^.]{0,30}?\d+(?:\.\d+)?\s?%",
]
UNSUPPORTED_PATTERNS = [
    (r"\brisk[\s-]?free\b(?!\s+from\s+default)", "Nothing market linked is risk free."),
    (r"\bno\s+risk\b", "All investments carry some risk."),
    (r"\bcannot\s+lose\b|\bcan't\s+lose\b|\bnever\s+(?:lose|fall|falls|go(?:es)?\s+down)\b",
     "Market linked investments can lose value."),
    (r"\balways\s+(?:beats?|outperforms?|grows?|rises?|goes\s+up|gives?)\b", "Past patterns do not guarantee the future."),
    (r"\b100\s?%\s+safe\b", "No investment is completely safe."),
]


def check_certainty(text: str) -> list[SafetyFlag]:
    flags = []
    for s in sentences(text):
        for pat in GUARANTEE_PATTERNS:
            m = re.search(pat, s, re.I)
            if m and not _negated(s, m.start()):
                flags.append(_flag(SafetyCategory.GUARANTEED_RETURN, H, s,
                                   "Use assumption based language, e.g. 'assuming an illustrative X% return, "
                                   "this may reach...'; returns are not guaranteed."))
                break
        for pat, why in UNSUPPORTED_PATTERNS:
            m = re.search(pat, s, re.I)
            if m and not _negated(s, m.start()):
                flags.append(_flag(SafetyCategory.UNSUPPORTED_CLAIM, H, s, why))
                break
    return flags


# ---------------------------------------------------------------------------
# C: risk mismatch
# ---------------------------------------------------------------------------
EQUITY_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%[^.]{0,40}?\bequit(?:y|ies)\b|\bequit(?:y|ies)\b[^.]{0,25}?(\d{1,3}(?:\.\d+)?)\s?%", re.I)


def max_equity_for(category: str | None, horizon_years: int | None) -> float | None:
    limits = [config.MAX_EQUITY_PCT[category]] if category in config.MAX_EQUITY_PCT else []
    if horizon_years is not None:
        if horizon_years < config.GUARDRAIL_CONSERVATIVE_BELOW_YEARS:
            limits.append(config.MAX_EQUITY_PCT["Conservative"])
        elif horizon_years <= config.GUARDRAIL_MODERATE_UP_TO_YEARS:
            limits.append(config.MAX_EQUITY_PCT["Moderate"])
    return min(limits) if limits else None


def check_risk_mismatch_text(text: str, category: str | None, horizon_years: int | None) -> list[SafetyFlag]:
    limit = max_equity_for(category, horizon_years)
    if limit is None:
        return []
    flags = []
    for s in sentences(text):
        for m in EQUITY_PCT.finditer(s):
            pct = float(m.group(1) or m.group(2))
            if pct > limit + 0.01:
                flags.append(_flag(SafetyCategory.RISK_MISMATCH, H, s,
                                   f"Equity of {pct:g}% exceeds the {limit:g}% limit for a "
                                   f"{category or ''} profile / {horizon_years} year horizon."))
                break
    return flags


ASSET_PCT_FWD = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%\s+(?:[a-z]+\s+){0,2}?(equit(?:y|ies)|debt|gold)\b", re.I)
ASSET_PCT_BACK = re.compile(r"\b(equit(?:y|ies)|debt|gold)\s+(?:at\s+|of\s+|is\s+|:\s*)?(\d{1,3}(?:\.\d+)?)\s?%", re.I)
LEANS_DEBT = re.compile(r"\b(lean\w*|tilt\w*|heav\w*|weight\w*|mostly|majority)\b[^.]{0,25}\bdebt\b", re.I)
LEANS_EQUITY = re.compile(r"\b(lean\w*|tilt\w*|heav\w*|weight\w*|mostly|majority)\b[^.]{0,25}\bequit(?:y|ies)\b", re.I)
CONSERVATIVE_MIX = re.compile(r"\b(conservative|lower[- ]equity)\b[^.]{0,25}\b(mix|allocation|portfolio)\b", re.I)


def check_allocation_claims(text: str, allocation: dict) -> list[SafetyFlag]:
    """The narrative must describe the allocation that was actually calculated.
    Catches e.g. '30% equity, 60% debt' when the plan is 60/30/10, or 'leaning toward debt'
    when equity is the largest part. (Found in the live Gemini evaluation, users U07 and U12.)"""
    actual = {"equity": allocation.get("equity_pct"), "debt": allocation.get("debt_pct"),
              "gold": allocation.get("gold_pct")}
    if any(v is None for v in actual.values()):
        return []
    eq, debt = actual["equity"], actual["debt"]
    flags = []
    for s in sentences(text):
        pairs = [(m.group(2), m.group(1)) for m in ASSET_PCT_FWD.finditer(s)]
        pairs += [(m.group(1), m.group(2)) for m in ASSET_PCT_BACK.finditer(s)]
        wrong = []
        for asset, pct in pairs:
            key = "equity" if asset.lower().startswith("equit") else asset.lower()
            if abs(float(pct) - actual[key]) > 0.5:
                wrong.append(f"{float(pct):g}% {key}")
        if wrong:
            flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                               f"States {', '.join(wrong)}, but the calculated allocation is "
                               f"{eq:g}% equity, {debt:g}% debt, {actual['gold']:g}% gold."))
            continue
        if (eq > debt and LEANS_DEBT.search(s)) or (debt > eq and LEANS_EQUITY.search(s)) or \
           (allocation.get("effective_category") not in (None, "Conservative") and CONSERVATIVE_MIX.search(s)):
            flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                               f"Describes the mix wrongly. The calculated allocation is {eq:g}% equity, "
                               f"{debt:g}% debt, {actual['gold']:g}% gold "
                               f"({allocation.get('effective_category') or 'unknown'} table)."))
    return flags


def check_allocation(allocation: dict) -> list[SafetyFlag]:
    """Defence in depth: the deterministic allocation must respect its own guardrail."""
    horizon = allocation.get("horizon_years")
    limit = max_equity_for(allocation.get("effective_category"), horizon)
    eq = allocation.get("equity_pct", 0)
    if limit is not None and eq > limit + 0.01:
        return [_flag(SafetyCategory.RISK_MISMATCH, H, f"Allocation has {eq:g}% equity",
                      f"Limit for this category/horizon is {limit:g}%.")]
    return []


# ---------------------------------------------------------------------------
# D: hallucination
# ---------------------------------------------------------------------------
GENERIC_PRODUCT_WORDS = set("""
the a an your this that my our equity debt mutual gold index liquid hybrid balanced gilt sectoral thematic
large mid small cap multi asset allocation systematic investment sip elss money market overnight arbitrage
retirement emergency growth income short long term duration corporate bond government sovereign
exchange traded fund funds etf etfs dynamic flexi value dividend yield solution oriented children's
national pension provident public ppf nps
""".split())
PRODUCT_NAME = re.compile(r"\b((?:[A-Z][\w&'.-]*\s+)(?:[A-Z0-9][\w&'.-]*\s+){0,4})(Fund|Funds|ETF|ETFs|Scheme|Bonds?)\b")
REGULATION = re.compile(r"\b(SEBI|RBI|AMFI|IRDAI|PFRDA|government|law|regulations?|rules?)\b[^.]{0,20}?"
                        r"\b(requires?|mandates?|mandatory|compulsory|must|stipulates?|obliges?)\b", re.I)
HISTORICAL = re.compile(r"\b(has|have|had)\s+(?:historically\s+)?(returned|delivered|given|generated|earned)\b[^.]{0,40}?\d+(?:\.\d+)?\s?%"
                        r"|\bhistorically\b[^.]{0,60}?\d+(?:\.\d+)?\s?%|\b(?:last|past)\s+\d+\s+years?\b[^.]{0,40}?\d+(?:\.\d+)?\s?%", re.I)
SOURCE_TAG = re.compile(r"\(\s*source[s]?\s*:\s*([^)]+)\)", re.I)
ACCORDING_TO = re.compile(r"\baccording to (?:the )?([A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,4})")
KNOWN_ORGS = {"SEBI", "AMFI", "NCFE", "Mutual Funds Sahi Hai"}


def check_products_and_claims(text: str, context_text: str = "", valid_sources: set[str] | None = None) -> list[SafetyFlag]:
    flags = []
    ctx_lower = context_text.lower()
    for s in sentences(text):
        for m in PRODUCT_NAME.finditer(s):
            words = [w.strip(".,'").lower() for w in m.group(1).split()]
            name = (m.group(1) + m.group(2)).strip()
            if any(w not in GENERIC_PRODUCT_WORDS for w in words) and name.lower() not in ctx_lower:
                flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                                   f"'{name}' looks like a specific product. The app must not name funds or products."))
                break
        m = REGULATION.search(s)
        if m and not _negated(s, m.start()) and s.lower()[:60] not in ctx_lower:
            flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                               "Regulatory claims must come from the knowledge base. Remove or cite a retrieved source."))
        if HISTORICAL.search(s) and s.lower()[:60] not in ctx_lower:
            flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                               "The app has no historical performance data. Remove historical return claims."))
        if valid_sources is not None:
            for tag in SOURCE_TAG.findall(s):
                for src in re.split(r"[,;]| and ", tag):
                    src = src.strip()
                    if src and src not in valid_sources and src not in KNOWN_ORGS:
                        flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                                           f"Source '{src}' was not retrieved. Cite only retrieved files."))
            for org in ACCORDING_TO.findall(s):
                if org not in KNOWN_ORGS and org not in valid_sources:
                    flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                                       f"'{org}' is not one of the retrieved sources."))
    return flags


# ---- numbers ---------------------------------------------------------------
UNIT = {"lakh": 1e5, "lakhs": 1e5, "l": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7}
RUPEE = re.compile(r"(?:₹|\bRs\.?|\bINR)\s?(-?[\d,]*\.?\d+)\s*(lakhs?|crores?|L|Cr)?\b"
                   r"|\b(\d+(?:\.\d+)?)\s*(lakhs?|crores?)\b", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
BARE_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*\.?\d*)")


def extract_amounts(text: str) -> list[tuple[float, bool, str]]:
    """(value in rupees, was_rounded_with_unit, matched text)"""
    out = []
    for m in RUPEE.finditer(text or ""):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        mult = UNIT.get((unit or "").lower(), 1)
        out.append((abs(value) * mult, mult > 1, m.group(0).strip()))
    return out


def extract_percents(text: str) -> list[float]:
    return [float(x) for x in PERCENT.findall(text or "")]


def collect_numbers(obj) -> set[float]:
    """All numbers inside nested facts (dicts/lists/models) and inside any strings."""
    found: set[float] = set()
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float)):
        found.add(abs(float(obj)))
    elif isinstance(obj, str):
        for v, _, _ in extract_amounts(obj):
            found.add(v)
        for n in BARE_NUMBER.findall(obj):
            try:
                found.add(float(n.replace(",", "")))
            except ValueError:
                pass
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= collect_numbers(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            found |= collect_numbers(v)
    return found


def _matches(value: float, allowed: Iterable[float], rel_tol: float, abs_tol: float) -> bool:
    return any(abs(value - a) <= max(abs_tol, rel_tol * abs(a)) for a in allowed)


def check_numbers(text: str, allowed: set[float]) -> list[SafetyFlag]:
    flags = []
    for s in sentences(text):
        for value, rounded, raw in extract_amounts(s):
            if value < 100:
                continue
            tol = config.SAFETY_ROUNDED_TOLERANCE if rounded else config.SAFETY_NUMBER_TOLERANCE
            if not _matches(value, allowed, tol, 1.0):
                flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                                   f"Amount '{raw}' does not match any calculated value or source."))
                break
        else:
            for pct in extract_percents(s):
                tol = 0.5 if pct.is_integer() else 0.05
                if not _matches(pct, allowed, 0.0, tol):
                    flags.append(_flag(SafetyCategory.HALLUCINATION, H, s,
                                       f"Percentage '{pct:g}%' does not match any assumption, calculation or source."))
                    break
    return flags


# ---------------------------------------------------------------------------
# E: missing assumptions
# ---------------------------------------------------------------------------
PROJECTION = re.compile(r"\b(reach|grow|grows|become|accumulate|projected|future value|corpus|target|will have|worth)\b", re.I)
ASSUMPTION = re.compile(r"\bassum|\billustrative\b|\bexpected (annual )?return\b|\d+(\.\d+)?\s?%\s*(a year|per year|annual|p\.a\.|return)", re.I)


def check_assumptions_stated(text: str) -> list[SafetyFlag]:
    has_projection = any(PROJECTION.search(s) and extract_amounts(s) for s in sentences(text))
    if has_projection and not ASSUMPTION.search(text or ""):
        return [_flag(SafetyCategory.MISSING_ASSUMPTION, H, text[:200],
                      "State the assumed annual return (and inflation, if used) behind any projected amount.")]
    return []


def check_plan_assumptions(assumptions: dict) -> list[SafetyFlag]:
    needed = ["expected_annual_return_pct", "annual_inflation_pct", "horizon_years", "contribution_frequency"]
    missing = [k for k in needed if assumptions.get(k) in (None, "")]
    if missing:
        return [_flag(SafetyCategory.MISSING_ASSUMPTION, H, f"Missing assumptions: {missing}",
                      "Show every assumption in the plan.")]
    return []


# ---------------------------------------------------------------------------
# F: escalation
# ---------------------------------------------------------------------------
ADVICE = re.compile(r"\b(advis[eo]r|professional|planner|expert)\b", re.I)
INVEST_MORE = re.compile(r"\b(increase|raise|add|top up|invest)\b[^.]{0,30}\b(sip|investment|invest|more)\b", re.I)


def stressed(ctx: dict) -> list[str]:
    reasons = []
    if (ctx.get("monthly_surplus") is not None and ctx["monthly_surplus"] < 0):
        reasons.append("negative monthly cash flow")
    if (ctx.get("debt_to_income_pct") or 0) > 40:
        reasons.append("EMIs above 40% of income")
    if (ctx.get("dependents") or 0) >= 3:
        reasons.append("three or more dependents")
    return reasons


def check_escalation_text(text: str, ctx: dict) -> list[SafetyFlag]:
    reasons = stressed(ctx)
    if reasons and INVEST_MORE.search(text or "") and not ADVICE.search(text or ""):
        return [_flag(SafetyCategory.ESCALATION, H, text[:200],
                      f"Situation is complex ({', '.join(reasons)}). Recommend consulting a SEBI registered "
                      "investment adviser before investing more.")]
    return []


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def llm_judge(text: str, facts: dict, context_chunks: list[RetrievedChunk], llm: LLMClient) -> list[SafetyFlag]:
    res = llm.structured(SafetyJudgeLLM, SAFETY_JUDGE_SYSTEM, SAFETY_JUDGE_USER.format(
        facts=json.dumps(facts, indent=1, default=str, ensure_ascii=False)[:6000],
        context=format_context(context_chunks)[:6000], text=text))
    return [SafetyFlag(category=i.category, severity=i.severity, text=i.text[:300],
                       suggestion=f"[LLM judge] {i.suggestion}") for i in res.issues]


def _dedupe(flags: list[SafetyFlag]) -> list[SafetyFlag]:
    seen, out = set(), []
    for f in flags:
        key = (f.category, f.text[:120])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_text(
    text: str,
    context: dict | None = None,
    context_chunks: list[RetrievedChunk] | None = None,
    allowed_numbers: set[float] | None = None,
    valid_sources: set[str] | None = None,
    llm: LLMClient | None = None,
    use_llm: bool | None = None,
) -> SafetyReport:
    """
    Check any AI written text.
    context        : user facts such as effective_category, horizon_years, monthly_surplus,
                     debt_to_income_pct, dependents (used for risk mismatch and escalation)
    allowed_numbers: numbers the text may use. None = skip the number check.
    valid_sources  : file names that were actually retrieved. None = skip the source check.
    """
    ctx = context or {}
    chunks = context_chunks or []
    ctx_text = "\n".join(c.text for c in chunks)
    flags = []
    flags += check_certainty(text)
    flags += check_risk_mismatch_text(text, ctx.get("effective_category"), ctx.get("horizon_years"))
    flags += check_products_and_claims(text, ctx_text, valid_sources)
    if allowed_numbers is not None:
        flags += check_numbers(text, allowed_numbers | collect_numbers(ctx_text))
    flags += check_assumptions_stated(text)
    flags += check_escalation_text(text, ctx)

    report = SafetyReport(flags=_dedupe(flags))
    use_llm = config.SAFETY_USE_LLM_JUDGE if use_llm is None else use_llm
    if llm is not None and use_llm and text.strip():
        try:
            report = SafetyReport(flags=_dedupe(report.flags + llm_judge(text, ctx, chunks, llm)),
                                  checked_by_llm=True)
        except LLMCallError:
            pass  # rules still stand; checked_by_llm stays False
    return report


def narrative_text(plan: FinalPlan) -> str:
    n = plan.narrative
    if n is None:
        return ""
    parts = [n.summary, n.allocation_explanation, *n.action_plan,
             *[f"{r.name}: {r.explanation}" for r in n.risk_explanations]]
    return "\n".join(p for p in parts if p)


def check_plan(plan: FinalPlan, context_chunks: list[RetrievedChunk] | None = None,
               llm: LLMClient | None = None, use_llm: bool | None = None) -> SafetyReport:
    """Check a complete plan: structure first, then the AI written narrative."""
    alloc = dict(plan.allocation, horizon_years=plan.goal.get("horizon_years"))
    flags = check_allocation(alloc) + check_plan_assumptions(plan.assumptions)

    text = narrative_text(plan)
    if not text:
        return SafetyReport(flags=_dedupe(flags))

    facts = {k: getattr(plan, k) for k in ("snapshot", "goal", "risk", "goal_calculation", "allocation",
                                           "assumptions", "scenarios")}
    ctx = {
        "effective_category": plan.allocation.get("effective_category"),
        "horizon_years": plan.goal.get("horizon_years"),
        "monthly_surplus": plan.snapshot.get("monthly_surplus"),
        "debt_to_income_pct": plan.snapshot.get("debt_to_income_pct"),
        "dependents": None,
    }
    chunks = context_chunks or []
    valid_sources = {c.source_file for c in chunks}
    report = check_text(text, ctx, chunks, collect_numbers(facts), valid_sources, llm, use_llm)
    flags += report.flags
    flags += check_allocation_claims(text, plan.allocation)

    if plan.professional_advice_recommended and not ADVICE.search(text):
        flags.append(_flag(SafetyCategory.ESCALATION, H, "Narrative does not recommend professional advice",
                           "Add an action to consult a SEBI registered investment adviser because: "
                           + "; ".join(plan.escalation_reasons)))
    return SafetyReport(flags=_dedupe(flags), checked_by_llm=report.checked_by_llm)


# ---------------------------------------------------------------------------
# Adapters for the LangGraph graphs
# ---------------------------------------------------------------------------
def make_plan_safety_fn(llm: LLMClient | None, use_llm: bool | None = None):
    def safety_fn(plan: FinalPlan, state: dict) -> SafetyReport:
        return check_plan(plan, state.get("context_chunks"), llm, use_llm)
    return safety_fn


def make_qa_safety_fn(llm: LLMClient | None, use_llm: bool | None = None):
    def safety_fn(text: str, state: dict) -> SafetyReport:
        pc = state.get("plan_context") or {}
        chunks = state.get("context_chunks") or []
        allowed = collect_numbers(pc) | collect_numbers(state.get("question", ""))
        for call in state.get("tool_calls") or []:
            allowed |= collect_numbers(call.output) | collect_numbers(call.inputs)
        ctx = {
            "effective_category": pc.get("effective_category"),
            "horizon_years": pc.get("horizon_years"),
            "monthly_surplus": pc.get("monthly_surplus"),
            "debt_to_income_pct": pc.get("debt_to_income_pct"),
            "dependents": pc.get("dependents"),
        }
        valid = {c.source_file for c in chunks}
        return check_text(text, ctx, chunks, allowed, valid if chunks else None, llm, use_llm)
    return safety_fn
