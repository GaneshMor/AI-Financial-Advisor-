# Evaluation results

**Configuration:** `{"mode": "offline", "model": null, "judge_model": null, "retriever": "keyword (BM25 fallback)", "rag_min_score": 0.3, "users": 24, "risk_repeats": 3, "timestamp": "2026-09-30T08:21:18"}`

| Metric | Result | Method | Notes |
|---|---|---|---|
| Calculation accuracy | 100.0% (10/10) | Tool output vs independent simulation ground truth, tolerance Rs 1 |  |
| Risk classification (rules) | 100.0% (30/30) | 6 boundary cases + 24 users, questionnaire score vs label | Deterministic by design; 100% expected. Consistency of the AI explanation is measured separately. |
| Retrieval hit@1 | 85.7% (6/7) | keyword (BM25 fallback); expected source ranked first |  |
| Retrieval hit@4 | 100.0% (7/7) | keyword (BM25 fallback); expected source in top 4 |  |
| Safety detection (rules only) | 100.0% (8/8) | 8 labelled texts: planted violations detected, clean text not blocked | Extra categories flagged beyond labels: 0; Rules were developed while viewing these 8 cases; use an unseen set for an unbiased figure |
| Risk explanation consistency (LLM) | Not run | Repeated risk agent runs | needs OPENAI_API_KEY |
| Goal classification accuracy (goal type) | Not run | 31 labelled goal texts | needs OPENAI_API_KEY |
| Tool selection accuracy | Not run | 8 labelled questions | needs OPENAI_API_KEY |
| Q&A routing accuracy | Not run | 8 labelled questions | needs OPENAI_API_KEY |
| RAG citation accuracy | Not run | 7 labelled questions | needs OPENAI_API_KEY |
| RAG groundedness | Not run | LLM judge | needs OPENAI_API_KEY |
| Plans generated without error | 100.0% (24/24) | Full LangGraph run for 24 users | Average time per plan: 0.0s |
| Disclaimer and assumptions present | 100.0% (24/24) | Every plan shows disclaimer, return, inflation, horizon and contribution frequency |  |
| Final plan text free of guaranteed-return language | Not run | Rule re-check of the final narrative (withheld text counts as clean) | needs OPENAI_API_KEY (offline plans contain no AI text) |
| Plans passing safety | Not run | Safety status passed (first try or after revision) | needs OPENAI_API_KEY (without it there is no AI text to check); Status counts: {'passed_structure_only': 24}; total revisions: 0 |
| Plan narrative groundedness | Not run | LLM judge: claims supported by plan facts + context | needs OPENAI_API_KEY |
| Hallucination rate (unsupported claims) | Not run | Unsupported / all judged claims (RAG answers + plan narratives) | needs OPENAI_API_KEY |

Rates are passed / total. For the hallucination rate, the count shown is UNSUPPORTED claims (lower is better).
