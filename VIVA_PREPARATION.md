# Viva Preparation: AI Personal Financial Advisor

Short, accurate answers tied to what this project actually does. File names are given so you can show the code if asked. Numbers come from the full run `evaluation/results/run_20260930_222538` (30 September 2026, Google Gemini `gemini-3.5-flash-lite`, 12 users, the same model as judge). Two figures are still **Pending** human work.

---

## A. Problem and motivation

**1. Why did you select this problem?**
Many people have goals (a house, a child's education, retirement) but cannot turn them into a monthly investment number, and do not see how inflation, time horizon and risk change that number. The problem needs both exact calculation and clear explanation, which makes it a good test of combining deterministic code with generative AI.

**2. Why use AI at all? A spreadsheet can do SIP maths.**
The maths is done by Python, like a spreadsheet. AI adds what a spreadsheet cannot: reading a goal written in plain language ("flat in Pune in 7 years, around 80 lakh"), explaining the result in simple words, answering follow up questions by choosing the right calculator, and grounding explanations in investor education material.

**3. What is the business impact?**
For platforms such as banks, AMCs, fintech apps and advisory firms: goal based onboarding at low cost, consistent explanations, and a way to spot users who need a human adviser (the escalation rules). For users: financial education personalised to their own numbers. The impact claimed is qualitative; the prototype has not been tested with real users.

## B. Agents and architecture

**4. What is an agent?**
A component that pursues a specific goal by deciding what to do next: which tool to call, what to write, or where to route, based on its input. In this project each agent has one role, its own prompt, and a Pydantic output schema.

**5. Difference between a chatbot and an agent?**
A chatbot maps a message to a reply. An agent can take actions: here the calculation agent chooses and calls Python tools, and the router chooses which agent handles a question. The system also has state, a fixed workflow and checks, not just a conversation.

**6. Why multiple agents instead of one?**
Separation of concerns. Each agent has a narrow prompt, so it is easier to test, evaluate and fix. The profile agent never gives advice; the risk agent cannot change the score; the safety agent checks the others. One large prompt doing everything is harder to control and to evaluate.

**7. Why not build a single chatbot?**
A single chatbot would do maths in text (unreliable), mix facts with guesses, and have no independent check. The multi-agent design keeps numbers deterministic, grounds facts in retrieved documents, and puts a safety check between the AI and the user.

**8. What is the role of each agent?**
Profile & Goal: cash flow and goal extraction. Risk: score and explanation. Portfolio: illustrative allocation and explanation. Calculation: runs tools (plan) and does tool calling (chat). RAG: grounded knowledge answers. Final Advisor: writes the plan from facts. Safety: checks all AI text. A router decides where each chat question goes.

**9. Why LangGraph?**
The workflow is a graph with a loop (safety → revise → advisor, at most 2 times) and branches (router, fatal error exits). LangGraph expresses that explicitly with typed state and conditional edges (`graph/plan_graph.py`, `graph/qa_graph.py`), and it can export the graph as a diagram. A simple chain cannot express the revision loop cleanly.

**10. Why does the Portfolio Agent run before the Calculation Agent?**
The required SIP depends on the expected return, and the expected return is the weighted return of the allocation (Moderate: 0.6×12 + 0.3×7 + 0.1×8 = 10.1%). So the allocation must exist first.

**11. What happens when an agent fails?**
Every graph node is wrapped by a guard. Invalid input or a calculation failure stops the plan with a clear error (no plan without numbers). An LLM failure (no key, timeout, bad JSON) is not fatal: the agent falls back, all numbers are still produced, and the missing explanation is labelled. Tests simulate each of these.

## C. Tool calling and calculations

**12. What is tool calling?**
The LLM receives tool descriptions and argument schemas, and instead of answering directly it returns a structured request such as `calculate_required_sip(target_amount=1000000, annual_return_pct=12, horizon_years=5)`. Python runs the function and sends the result back; the LLM then writes the answer (`agents/calculation_agent.py`, `tools/lc_tools.py`).

**13. Why use Python tools instead of the LLM for calculations?**
LLMs predict text; they can make arithmetic errors and give different answers to the same question. Financial numbers must be exact and repeatable. Python gives the same answer every time and can be unit tested against an independent ground truth.

**14. What assumptions are used in the SIP calculation?**
Contribution at the start of each month; monthly rate = annual return ÷ 12; monthly compounding; the current SIP is dedicated to this one goal; returns and inflation are constant illustrative assumptions. Formula: FV = P × [((1+r)^n − 1)/r] × (1+r), with r = 0 handled separately.

**15. What happens if the expected return changes?**
The required SIP moves in the opposite direction. For example, ₹10 lakh in 10 years needs a larger SIP at 8% than at 10%. The What If page and the scenario tool show this; the standard scenarios use the assumed return ±2 points and are labelled hypothetical.

**16. Why is inflation important?**
A goal priced in today's rupees costs more in future. At 6% inflation, ₹25 lakh today becomes about ₹35.5 lakh in 6 years. Planning without inflation would understate both the target and the required SIP.

**17. How did you check the calculations are correct?**
Ground truth is computed in `scripts/build_datasets.py` without using the tools: the account is simulated month by month and the required SIP is found by bisection. The tools must match within ₹1. Result: 100% (10/10), in both the offline and the full run.

## D. Risk profiling and allocation

**18. How does the risk profiling work?**
7 questions (horizon, reaction to a 20% fall, income stability, EMI share, emergency fund, experience, preference), 1 to 3 points each, total 7 to 21. Bands: 7–11 Conservative, 12–16 Moderate, 17–21 Aggressive. The LLM only explains the result.

**19. Why is your risk profile not a regulated suitability assessment?**
It is a simple academic questionnaire. It does not include full KYC, a complete financial and tax picture, or regulatory documentation, and it has not been validated statistically. The app says this on every risk result.

**20. What is asset allocation?**
Dividing investments across asset classes (equity, debt, gold) according to goals, horizon and risk appetite. Here it is illustrative: Conservative 30/60/10, Moderate 60/30/10, Aggressive 75/15/10.

**21. What is diversification?**
Spreading money across different investments so one bad outcome has less effect. It reduces risk but does not remove market risk or guarantee against loss, as SEBI and AMFI material states (`rag/knowledge_base/diversification.md`).

**22. How did you handle financial risk in the design?**
A time horizon guardrail (goals under 3 years use Conservative; 3 to 5 years are capped at Moderate), an emergency reserve check before investing, 100% debt for emergency fund goals, a risk mismatch check in the Safety Agent, and escalation rules that recommend a SEBI registered adviser.

## E. RAG and hallucination

**23. What is RAG?**
Retrieval-Augmented Generation: before the LLM answers, relevant passages are retrieved from a trusted knowledge base and given to it as context, and it is told to answer only from them.

**24. Why is RAG required here?**
Without it, the LLM answers from training data that may be outdated, non Indian, or wrong, and it cannot show a source. RAG ties explanations to SEBI, AMFI and NCFE material and lets the user see which file was used.

**25. How does your RAG pipeline work?**
13 markdown files → split by section (max 900 characters, 70 chunks) → embeddings (Google `gemini-embedding-001` in our runs) → FAISS (cosine). For a question: embed → top 4 chunks with score ≥ 0.30 → prompt "answer only from context, cite the file" → LLM → remove any cited source not actually retrieved. Without a key, BM25 keyword search is used and labelled.

**26. How did you reduce hallucination?**
Five layers: (1) numbers only from Python; (2) facts only from retrieved context; (3) structured outputs validated by Pydantic; (4) the Safety Agent flags unmatched numbers, named products, invented regulations, historical returns and unretrieved sources; (5) text that still fails after 2 revisions is withheld. Measured hallucination rate (LLM judge): 17.6% (34/193 claims). Be ready to explain it: we read all 34. Five were real errors, where the narrative described the mix as 30% equity / 60% debt or "leaning toward debt" for 2 users whose calculated mix was 60/30/10. The Safety Agent missed them because 30 and 60 exist in the plan facts, just for a different asset. We then added `check_allocation_claims`, which caught all 5 and flagged none of the other 164 plan claims when replayed. Most of the other 29 are correct general advice the judge could not match to a fact; the manual review will confirm.

**27. What happens if the LLM gives incorrect information?**
If it is a number, the safety check flags it because it does not match any calculated value. If it is a promise, product name, regulation or unretrieved source, the rules flag it. Flagged plan text is rewritten or withheld; flagged chat answers are withheld. Subtle wrong statements depend on the LLM judge and may pass; that is a stated limitation.

## F. Responsible AI, evaluation, privacy

**28. What is responsible AI in this project?**
Transparency (every assumption shown, sources cited), reliability (deterministic numbers, tested fallbacks), safety (no guarantees, no products, no invented facts), accountability (session log, agent trace, safety status), human oversight (escalation to a professional), and privacy.

**29. How did you evaluate the system?**
`python -m evaluation.evaluate` runs 54 labelled cases and 24 synthetic users: calculation accuracy, risk classification, goal classification, tool selection, routing, retrieval hit rate, citation accuracy, groundedness and hallucination rate (LLM judge plus human review), safety detection, and plan quality. Key results (Gemini, 12 users): calculation 100% (10/10); goal parsing 100% (7/7); goal type 93.5% (29/31); tool selection 100% (8/8); routing 100% (8/8); retrieval hit@1 100% (7/7, FAISS); RAG citation 100% (7/7); RAG groundedness 100%; plan groundedness 79.9%; 11 of 12 plans passed safety (7 first time, 4 after one revision, 1 withheld); risk explanation consistency 100% (5/5).

**30. How do you know the LLM judge is right?**
We do not assume it is. Every judged claim is written to `manual_review.csv`; a human marks them and `score_manual_review` reports judge–human agreement. Agreement: **Pending** (manual review not yet done). Note that the judge and the generator are the same model in our run, which is a known weakness. A different judge model can be set with `JUDGE_MODEL_NAME` to reduce self-grading bias.

**31. Isn't 100% safety detection suspicious?**
Yes. The rules were written while looking at those 8 cases, so it is an optimistic figure. For an honest figure, a teammate writes an unseen set in `data/safety_unseen.csv`: **Pending**.

**32. How is privacy handled?**
No real personal data: the 24 users are synthetic. The session log records field names, never amounts. Nothing is stored after the browser session. The API key stays in `.env` and is never logged. The data is sent to the LLM provider for processing, which a production system would have to disclose and cover by agreement.

**33. What are the limitations?**
One goal per plan; constant illustrative returns; no market data or Monte Carlo; a simple questionnaire; a small knowledge base written by the team; rule based safety misses subtle claims; no tax or product advice; small evaluation set labelled by the team.

**34. How can this be deployed commercially?**
Host the Streamlit app (or rebuild the UI in a web framework) behind authentication; move storage to an encrypted database with consent; add monitoring of safety flags and costs; keep the knowledge base under compliance review; and operate under the applicable SEBI framework for investment advice or keep the product strictly educational.

**35. What would you improve in version 2?**
Multiple goals, Monte Carlo ranges, tax aware planning, a larger and regularly refreshed knowledge base, hybrid retrieval with reranking, saved profiles, Hindi and regional languages, and a larger independently labelled evaluation set.

**36. Which LLM did you use, and why?**
Google Gemini `gemini-3.5-flash-lite` for chat and `gemini-embedding-001` for embeddings, called through Gemini's OpenAI-compatible endpoint, because it has a free tier. The code is provider independent: the model, endpoint and key come from `.env` (`MODEL_NAME`, `LLM_BASE_URL`), so it also runs on OpenAI. On the free tier we pause 4 seconds between calls and retry on rate limits, which is why a plan takes about 48 seconds.

**37. Did your evaluation find any real bug?**
Yes. In the first full run the narrative for 2 users described the wrong mix (30% equity / 60% debt instead of 60/30/10) and the Safety Agent passed it. We added `check_allocation_claims` and tests for it. The full run in the report is from before that fix; we checked the fix by replaying it on all 169 plan claims from that run (5 caught, 0 false alarms). We also found that the tests were accidentally calling the real API when a key was present; `tests/conftest.py` now blocks that.

---

## Quick numbers to remember (sample user U02, default assumptions)

| Item | Value |
|---|---|
| Goal | House, ₹25 lakh today, 6 years |
| Inflation adjusted target | ₹35,46,298 |
| Required SIP / current SIP | ₹28,360 / ₹15,000 |
| Risk score | 16 → Moderate → 60/30/10, assumed return 10.1% |
| Tests | 338 automated tests |
| Knowledge base | 13 files, 70 chunks |
| Labelled cases | 54 (10 calculation, 7 goal parsing, 6 risk, 8 tool, 7 RAG, 8 safety, 8 routing) |
