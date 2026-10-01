# System Architecture

Rendered images of these diagrams are in `docs/diagrams/`. The Mermaid source below renders on GitHub or at https://mermaid.live.

## 1. Layered architecture

```mermaid
flowchart TB
    U([User])
    subgraph UI["1. Streamlit Dashboard  (app.py, ui/)"]
        direction LR
        P1[Profile / Goal / Risk forms] ~~~ P2[Analysis, Goal, Portfolio, What If] ~~~ P3[AI Advisor chat] ~~~ P4[Final Report + downloads]
    end
    subgraph ORCH["2. Orchestrator: LangGraph  (graph/)"]
        direction LR
        G1[Plan graph] ~~~ G2[Q&A graph with router]
    end
    subgraph AGENTS["3. Agents  (agents/)"]
        direction LR
        A1[Profile & Goal] ~~~ A2[Risk Profiling] ~~~ A3[Portfolio Allocation] ~~~ A4[Financial Calculation] ~~~ A5[RAG Knowledge] ~~~ A6[Final Advisor] ~~~ A7[Safety]
    end
    subgraph TOOLS["4a. Python tools  (tools/)  NUMBERS"]
        direction LR
        T1[SIP] ~~~ T2[Future value] ~~~ T3[Inflation] ~~~ T4[Goal gap] ~~~ T5[Health] ~~~ T6[Scenario] ~~~ T7[Allocation rules]
    end
    LLM[("4b. OpenAI chat model via LangChain  WORDS")]
    subgraph KB["4c. Knowledge layer  (rag/)  FACTS"]
        direction LR
        K1[13 files from SEBI, AMFI, NCFE material] ~~~ K2[FAISS + OpenAI embeddings] ~~~ K3[BM25 fallback]
    end
    U --> UI --> ORCH --> AGENTS
    AGENTS --> TOOLS
    AGENTS --> LLM
    AGENTS --> KB
```

**Core rule:** numbers come from Python tools, words come from the LLM, facts come from the knowledge base or the user. The Safety Agent enforces this.

## 2. Plan graph (LangGraph)

```mermaid
flowchart LR
    S((start)) --> V[validate]
    V -->|invalid| E((end))
    V --> PR[profile] --> RI[risk] --> PO[portfolio] --> CA[calculation] --> KN[knowledge] --> AD[advisor] --> SA[safety]
    SA -->|HIGH flag and revisions < 2| RV[revise]
    RV -->|flags as feedback| AD
    SA -->|clean, or 2 revisions used| FI[finalize] --> E2((end))
```

* Every node is wrapped by a guard: an unexpected exception is recorded and the graph stops cleanly.
* Validation and calculation failures are fatal (no plan without numbers). LLM and retrieval failures are not: agents fall back and the plan completes.
* If the narrative still fails safety after 2 revisions, `finalize` withholds it and keeps the numbers.

## 3. Q&A graph (LangGraph)

```mermaid
flowchart LR
    S((question)) --> R{router LLM}
    R -->|calculation| C[Calculation Agent<br/>LLM picks tool, Python runs it]
    R -->|knowledge| K[RAG Agent<br/>knowledge base + your plan]
    R -->|out of scope| O[fixed polite refusal]
    C --> SF[safety]
    K --> SF
    SF -->|HIGH flag| W[answer withheld with reason]
    SF -->|clean| A((answer))
    O --> A
```

## 4. Data flow

```mermaid
flowchart TB
    IN[Form inputs + goal text + 7 answers] --> VAL[Pydantic validation<br/>UserProfile, GoalInput]
    VAL --> PA[ProfileAgentOutput<br/>surplus, discrepancies, priority]
    PA --> RA[RiskResult<br/>score 7-21, category]
    RA --> AL[AllocationResult<br/>equity/debt/gold, guardrail, weighted return]
    AL --> CO[CalculationAgentOutput<br/>target, required SIP, gap, health, scenarios]
    CO --> CTX[RetrievedChunks<br/>with source file]
    CTX --> FP[FinalPlan<br/>10 sections + narrative]
    FP --> SR[SafetyReport<br/>flags, severity]
    SR --> OUT[Dashboard + Markdown/HTML report]
```

## 5. Tool flow

| Tool | Called by | Input | Output |
|---|---|---|---|
| `financial_health` | Profile, Calculation | income, expenses, EMI, SIP, emergency fund | surplus, savings rate, debt to income, emergency months |
| `compute_allocation` | Portfolio | risk category, horizon, goal type | equity/debt/gold %, guardrail, weighted return |
| `inflation_adjusted_value` | Calculation (via `plan_goal`) | amount, inflation %, years | future cost |
| `lumpsum_future_value`, `sip_future_value` | Calculation | amount or SIP, return %, months | future value |
| `required_monthly_sip` | Calculation | target, return %, months | monthly SIP |
| `goal_gap` | Calculation | required SIP, current SIP, free surplus | gap, status, affordability |
| `standard_scenarios`, `run_scenario` | Calculation, What If page | base inputs + overrides | scenario rows |
| 6 LangChain wrappers (`tools/lc_tools.py`) | Q&A Calculation Agent | chosen by the LLM | tool result JSON |

## 6. RAG flow

```mermaid
flowchart LR
    subgraph Ingest["python -m rag.ingest"]
        MD[13 .md files with source header] --> CH[Split by section<br/>max 900 chars, paragraph overlap] --> EM[OpenAI embeddings] --> IX[(FAISS index<br/>cosine similarity)]
    end
    subgraph Query
        Q[question] --> QE[embed] --> SE[top 4, score >= 0.30] --> PR[prompt: answer only from context] --> LLM --> CK[drop any cited source<br/>not retrieved] --> ANS[answer + sources]
    end
    IX --> SE
```

When there is no index or no API key, the retriever falls back to BM25 keyword search over the same chunks and says so in the UI.
