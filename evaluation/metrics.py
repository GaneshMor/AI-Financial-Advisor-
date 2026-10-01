"""
Pure scoring functions for the evaluation (no LLM, no I/O), so the scoring
itself can be unit tested.
"""

from dataclasses import dataclass, field


@dataclass
class Metric:
    name: str
    passed: int = 0
    total: int = 0
    method: str = ""
    notes: list[str] = field(default_factory=list)
    status: str = "run"          # "run" | "not_run"

    @property
    def rate(self) -> float | None:
        return None if self.total == 0 else self.passed / self.total

    def add(self, ok: bool) -> None:
        self.total += 1
        self.passed += int(bool(ok))

    def as_row(self) -> dict:
        return {"metric": self.name, "result": self.display(), "passed": self.passed, "total": self.total,
                "method": self.method, "status": self.status, "notes": "; ".join(self.notes)}

    def display(self) -> str:
        if self.status != "run":
            return "Not run"
        if self.total == 0:
            return "No cases"
        return f"{self.rate * 100:.1f}% ({self.passed}/{self.total})"


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
CALC_FIELDS = ("target_future_value", "fv_current_goal_savings", "required_monthly_sip",
               "projected_value_current_plan")


def score_calculation(result: dict, expected: dict, tol: float = 1.0) -> tuple[bool, list[str]]:
    """result = plan_goal(...).to_dict(); expected = ground truth row. Tolerance in rupees."""
    errors = [f"{k}: got {result[k]:.2f}, expected {expected[k]:.2f}"
              for k in CALC_FIELDS if abs(result[k] - expected[k]) > tol]
    if result["gap"]["status"] != expected["status"]:
        errors.append(f"status: got {result['gap']['status']}, expected {expected['status']}")
    return not errors, errors


# ---------------------------------------------------------------------------
# Goal parsing
# ---------------------------------------------------------------------------
PARSE_FIELDS = ("goal_type", "goal_amount", "horizon_years", "amount_is_present_value")


def score_goal_parse(got: dict, expected: dict, amount_tol: float = 0.01) -> tuple[bool, dict[str, bool]]:
    """Compare only the fields the label states (None = not stated = skipped)."""
    per_field: dict[str, bool] = {}
    for f in PARSE_FIELDS:
        exp = expected.get(f)
        if exp is None:
            continue
        g = got.get(f)
        if f == "goal_amount":
            per_field[f] = g is not None and abs(float(g) - float(exp)) <= amount_tol * float(exp)
        else:
            per_field[f] = g == exp
    if "missing_fields" in expected:
        per_field["missing_fields"] = set(expected["missing_fields"]) <= set(got.get("missing_fields") or [])
        for f in expected["missing_fields"]:
            per_field[f"{f}_is_null"] = got.get(f) is None
    return all(per_field.values()), per_field


# ---------------------------------------------------------------------------
# Retrieval, tools, routing
# ---------------------------------------------------------------------------
def hit_at_k(retrieved_sources: list[str], expected: str, k: int) -> bool:
    return expected in retrieved_sources[:k]


def score_tool_selection(called: list[str], expected: str) -> tuple[bool, bool]:
    """(expected tool called at all, expected tool called first)"""
    return expected in called, bool(called) and called[0] == expected


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def score_safety_case(detected: set[str], expected: set[str], high_detected: bool) -> tuple[bool, set[str]]:
    """
    Planted violation: pass if every expected category is detected.
    Clean text (expected empty): pass if there is no HIGH flag.
    Returns (passed, extra categories detected beyond the label).
    """
    extra = detected - expected
    if not expected:
        return not high_detected, extra
    return expected <= detected, extra


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------
def groundedness(claims: list[dict]) -> tuple[int, int]:
    """(supported, total) from judge output [{claim, supported}]."""
    return sum(1 for c in claims if c["supported"]), len(claims)


def summary_table(metrics: list[Metric]) -> str:
    lines = ["| Metric | Result | Method | Notes |", "|---|---|---|---|"]
    for m in metrics:
        lines.append(f"| {m.name} | {m.display()} | {m.method} | {'; '.join(m.notes)} |")
    return "\n".join(lines)
