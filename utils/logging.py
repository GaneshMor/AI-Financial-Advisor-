"""
Session level agent logging.

For each step we record: timestamp, agent, action, tool called, input FIELD
NAMES (never the values), a short output summary, status and safety flags.

Privacy rule: financial values are not written to the log. Only the names of
the inputs are kept, so the log shows WHAT was used, not the user's numbers.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LogEvent:
    timestamp: str
    session_id: str
    agent: str
    action: str
    tool: str | None
    input_fields: list[str]
    output_summary: str
    status: str                     # "ok" | "fallback" | "error"
    safety_flags: list[str] = field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None


class SessionLog:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.events: list[LogEvent] = []

    def record(
        self,
        agent: str,
        action: str,
        *,
        tool: str | None = None,
        inputs: dict | None = None,
        output_summary: str = "",
        status: str = "ok",
        safety_flags: list[str] | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self.events.append(
            LogEvent(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                session_id=self.session_id,
                agent=agent,
                action=action,
                tool=tool,
                input_fields=sorted(inputs.keys()) if inputs else [],
                output_summary=output_summary[:300],
                status=status,
                safety_flags=list(safety_flags or []),
                error=(error or None) and error[:300],
                duration_ms=None if duration_ms is None else round(duration_ms, 1),
            )
        )

    def to_records(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def save_jsonl(self, directory: str | Path = "logs") -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"session_{self.session_id}.jsonl"
        with file.open("a", encoding="utf-8") as f:
            for rec in self.to_records():
                f.write(json.dumps(rec) + "\n")
        return file


class Timer:
    """Usage:  with Timer() as t: ...   then read t.ms (elapsed milliseconds)."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000
        return False
