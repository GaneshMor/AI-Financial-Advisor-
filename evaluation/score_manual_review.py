"""
Agreement between the LLM judge and a human reviewer.

1. Open evaluation/results/run_<...>/manual_review.csv
2. Fill the human_supported column with TRUE or FALSE for at least 30 rows
3. python -m evaluation.score_manual_review evaluation/results/run_<...>/manual_review.csv
"""

import csv
import sys


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_supported"].strip()]
    if not rows:
        raise SystemExit("No rows have human_supported filled in yet.")
    truth = lambda v: v.strip().lower() in {"true", "1", "yes", "y"}  # noqa: E731
    agree = sum(truth(r["judge_supported"]) == truth(r["human_supported"]) for r in rows)
    human_unsupported = sum(not truth(r["human_supported"]) for r in rows)
    print(f"Rows reviewed: {len(rows)}")
    print(f"Judge-human agreement: {agree / len(rows) * 100:.1f}% ({agree}/{len(rows)})")
    print(f"Hallucination rate by human review: {human_unsupported / len(rows) * 100:.1f}% "
          f"({human_unsupported}/{len(rows)} claims unsupported)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manual_review.csv")
