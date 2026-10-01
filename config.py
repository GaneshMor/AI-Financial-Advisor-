"""
Central configuration for the AI Personal Financial Advisor.

Everything here is either:
  1. read from the .env file (API keys, model names), or
  2. an ILLUSTRATIVE ASSUMPTION used by the prototype.

None of the return or inflation numbers below are forecasts or historical
figures. They are round numbers chosen for demonstration and can be changed
by the user in the app.
"""

import os

# python-dotenv is optional so that the calculation tools still work
# even if the package is not installed yet.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# LLM settings (used from Phase 4 onwards). Never hard code keys here.
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
# Optional: any OpenAI-compatible provider, e.g. a free tier.
#   Google Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
#   Groq:          https://api.groq.com/openai/v1
# Leave empty for OpenAI itself.
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL") or None
MODEL_NAME: str = os.getenv("MODEL_NAME") or "gpt-4o-mini"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
# Evaluation only: the model that judges groundedness. A different model reduces self-grading bias.
JUDGE_MODEL_NAME: str = os.getenv("JUDGE_MODEL_NAME") or MODEL_NAME
LLM_TEMPERATURE: float = 0.0          # 0 = most repeatable output
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS") or 60)
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES") or 2)       # free tiers: use 6 (retries on rate limits)
LLM_REQUEST_DELAY_SECONDS: float = float(os.getenv("LLM_REQUEST_DELAY_SECONDS") or 0)  # free tiers: e.g. 4
QA_MAX_TOOL_ROUNDS: int = 3           # tool calling loop limit for the Q&A agent

# ---------------------------------------------------------------------------
# Illustrative assumptions (shown to the user every time they are used)
# ---------------------------------------------------------------------------
ASSUMPTION_LABEL = "Illustrative assumption. Not a forecast or historical figure."

ASSUMED_ANNUAL_RETURNS_PCT: dict[str, float] = {
    "equity": 12.0,
    "debt": 7.0,
    "gold": 8.0,
}
DEFAULT_INFLATION_PCT: float = 6.0

# Illustrative allocation tables (percent of the monthly investment)
ALLOCATION_TABLES: dict[str, dict[str, float]] = {
    "Conservative": {"equity": 30.0, "debt": 60.0, "gold": 10.0},
    "Moderate": {"equity": 60.0, "debt": 30.0, "gold": 10.0},
    "Aggressive": {"equity": 75.0, "debt": 15.0, "gold": 10.0},
}
# Emergency fund goals must stay liquid and stable
EMERGENCY_GOAL_ALLOCATION: dict[str, float] = {"equity": 0.0, "debt": 100.0, "gold": 0.0}
# Time horizon guardrail: horizon < 3 years -> Conservative; 3 to 5 years -> at most Moderate
GUARDRAIL_CONSERVATIVE_BELOW_YEARS: int = 3
GUARDRAIL_MODERATE_UP_TO_YEARS: int = 5
EMERGENCY_RESERVE_MONTHS: int = 6

# Contribution frequency used by every SIP calculation
CONTRIBUTION_FREQUENCY = "Monthly, at the start of each month"

# ---------------------------------------------------------------------------
# Input limits (used by the validation helpers)
# ---------------------------------------------------------------------------
MIN_RETURN_PCT: float = 0.0
MAX_RETURN_PCT: float = 50.0
MIN_INFLATION_PCT: float = 0.0
MAX_INFLATION_PCT: float = 30.0
MAX_HORIZON_YEARS: int = 50

# Gaps smaller than this (in rupees) are treated as "on track"
ON_TRACK_TOLERANCE_INR: float = 1.0

# ---------------------------------------------------------------------------
# Financial health thresholds. PROTOTYPE INDICATORS, not universal rules.
# ---------------------------------------------------------------------------
HEALTH_THRESHOLDS = {
    # savings rate %: >= good -> "Healthy", >= fair -> "Moderate", else "Low"
    "savings_rate_pct": {"good": 20.0, "fair": 10.0},
    # debt to income %: <= good -> "Comfortable", <= fair -> "Elevated", else "High"
    "debt_to_income_pct": {"good": 30.0, "fair": 40.0},
    # emergency fund months: >= good -> "Adequate", >= fair -> "Partial", else "Low"
    "emergency_fund_months": {"good": 6.0, "fair": 3.0},
}
# ---------------------------------------------------------------------------
# Knowledge base files (Phase 6 must create exactly these in rag/knowledge_base/)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_FILES = [
    "sip.md", "compounding.md", "inflation.md", "asset_allocation.md",
    "diversification.md", "equity_risk.md", "debt_basics.md", "gold.md",
    "emergency_fund.md", "risk_return.md", "mutual_fund_basics.md",
    "risk_disclosures.md", "financial_planning_basics.md",
]

from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "rag" / "knowledge_base"
RAG_INDEX_DIR = PROJECT_ROOT / "rag" / "index"
RAG_TOP_K: int = 4
RAG_CHUNK_CHARS: int = 900        # max characters per chunk
# Minimum cosine similarity for a FAISS hit to count as relevant. Starting value;
# tune it in Phase 9 using the rag_retrieval test cases.
RAG_MIN_SCORE: float = 0.30

# Safety Agent
SAFETY_USE_LLM_JUDGE: bool = True        # rules always run; the LLM judge adds a second check
SAFETY_NUMBER_TOLERANCE: float = 0.005   # 0.5% for exact rupee figures
SAFETY_ROUNDED_TOLERANCE: float = 0.02   # 2% for rounded figures like "38 lakh"
MAX_EQUITY_PCT = {"Conservative": 30.0, "Moderate": 60.0, "Aggressive": 75.0}

HEALTH_DISCLAIMER = (
    "Prototype indicators based on common rules of thumb. "
    "They are not universal financial rules."
)
