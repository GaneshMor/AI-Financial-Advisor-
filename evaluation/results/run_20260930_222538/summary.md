# Evaluation results

**Configuration:** `{"mode": "full", "model": "gemini-3.5-flash-lite", "judge_model": "gemini-3.5-flash-lite", "retriever": "faiss", "rag_min_score": 0.3, "users": 12, "risk_repeats": 3, "timestamp": "2026-09-30T22:25:38"}`

| Metric | Result | Method | Notes |
|---|---|---|---|
| Calculation accuracy | 100.0% (10/10) | Tool output vs independent simulation ground truth, tolerance Rs 1 |  |
| Risk classification (rules) | 100.0% (30/30) | 6 boundary cases + 24 users, questionnaire score vs label | Deterministic by design; 100% expected. Consistency of the AI explanation is measured separately. |
| Retrieval hit@1 | 100.0% (7/7) | faiss; expected source ranked first |  |
| Retrieval hit@4 | 100.0% (7/7) | faiss; expected source in top 4 |  |
| Safety detection (rules only) | 100.0% (8/8) | 8 labelled texts: planted violations detected, clean text not blocked | Extra categories flagged beyond labels: 0; Rules were developed while viewing these 8 cases; use an unseen set for an unbiased figure |
| Safety detection (rules + LLM judge) | 100.0% (8/8) | 8 labelled texts: planted violations detected, clean text not blocked | Extra categories flagged beyond labels: 4 |
| Risk explanation consistency (LLM) | 100.0% (5/5) | Risk agent run 3x on 5 users; pass if the category never changes and every explanation names the correct category |  |
| Goal parsing: all stated fields correct | 100.0% (7/7) | 7 labelled free text goals |  |
| Goal classification accuracy (goal type) | 93.5% (29/31) | 7 labelled goals + 24 user goal descriptions |  |
| Tool selection accuracy | 100.0% (8/8) | 8 labelled questions; expected tool among the tools called |  |
| Tool selection (first call correct) | 100.0% (8/8) | Expected tool is the first tool called |  |
| Q&A routing accuracy | 100.0% (8/8) | 8 labelled questions through the LangGraph router |  |
| RAG citation accuracy | 100.0% (7/7) | Answer is answerable and cites the expected knowledge file | Invented sources caught and removed: 0 |
| RAG groundedness | 100.0% (24/24) | LLM judge: claims supported by retrieved chunks / all claims |  |
| Plans generated without error | 100.0% (12/12) | Full LangGraph run for 12 users | Average time per plan: 48.3s |
| Disclaimer and assumptions present | 100.0% (12/12) | Every plan shows disclaimer, return, inflation, horizon and contribution frequency |  |
| Final plan text free of guaranteed-return language | 100.0% (12/12) | Rule re-check of the final narrative (withheld text counts as clean) |  |
| Plans passing safety | 91.7% (11/12) | Safety status passed (first try or after revision) | Status counts: {'passed': 11, 'failed_after_revisions': 1}; total revisions: 6 |
| Plan narrative groundedness | 79.9% (135/169) | LLM judge: claims supported by plan facts + context |  |
| Hallucination rate (unsupported claims) | 17.6% (34/193) | Unsupported / all judged claims (RAG answers + plan narratives) | LOWER is better. Judge is an LLM; verify with manual_review.csv |

Rates are passed / total. For the hallucination rate, the count shown is UNSUPPORTED claims (lower is better).
