# FinBrain

## Multi-Agent Financial Research System

A production-oriented AI agent workflow for automated equity research, combining structured data retrieval, RAG-based industry knowledge, multi-agent reasoning with critic verification, and deterministic guardrails — designed for reliability over generation.

---

## Motivation

LLM-based financial analysis tools typically suffer from four failure modes:

1. **Hallucinated data** — LLMs fabricate PE ratios, revenue figures, and financial metrics
2. **Missing domain context** — Generic models lack industry-specific valuation frameworks
3. **Inconsistent reasoning** — The same stock analyzed twice produces contradictory conclusions
4. **No verification** — Outputs are unchecked for mathematical consistency or logical coherence

FinBrain addresses each by decomposing financial research into specialized components: deterministic calculation for numbers, RAG for domain knowledge, multi-agent workflow for structured reasoning, and a dual-layer audit system (code + LLM) for verification.

---

## Architecture

```
User Query ("分析长电科技600584")
    │
    ▼
Intent Router ──→ Chat / Deep Analysis / Phantom Hunter
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Research Pipeline (LangGraph StateGraph)            │
│                                                      │
│  Data Agent → Classifier → Analyst → Valuation      │
│   (parallel)    (code)      (LLM)     (LLM)          │
│       │                                              │
│       └─ data missing? ──→ re-fetch once            │
│                                                      │
│  → Critics(3-way parallel) ──→ Repair? ──→ Reporter │
│     Logic/Financial/Industry   (LLM)    (LLM)        │
│       │                            │                 │
│       └─ no issues? ──→ skip Repair ──┘             │
│                                                      │
│  → Audit Engine (code)                               │
│     · score consistency · data unit validation       │
│     · dilution correction · scenario monotonicity    │
│     · FCF warning injection · structured output      │
└─────────────────────────────────────────────────────┘
    │
    ▼
Structured Report + Execution Trace + Audit Summary
```

**Key design principle**: Deterministic computation (financial metrics, scoring, valuation, verification) is handled by code. Probabilistic reasoning (investment thesis, competitive analysis, narrative generation) is delegated to LLMs. The boundary between them is explicit and enforced.

---

## Agent Workflow

### Data Agent (Parallel, no LLM)
Fetches financial statements, valuation metrics, stock prices, industry classification, and announcements concurrently via `ThreadPoolExecutor`. Computes TTM EPS, PE/PB, growth trends, and cash flow quality scores deterministically. Injects `[INDUSTRY]` and `[TOOLS]` headers for downstream traceability. Typical latency: ~5 seconds for 6 data sources.

### Analyst Agent (LLM)
Receives structured financial data + RAG-retrieved industry templates. Follows a 10-step analytical framework: company classification → business model → moat assessment → industry cycle → financial verification → growth logic → market expectations → scenario valuation → catalysts → investment decision. Outputs structured JSON.

### Valuation Agent (LLM)
Classifies the company stage (mature cyclical, cyclical recovery, stable growth, hyper-growth) and recommends appropriate valuation frameworks. For a company like 长电科技 (semiconductor packaging), it identifies "cyclical recovery with AI growth overlay" and recommends normalized-earnings PE + EV/EBITDA + PEG, explicitly flagging that static PE on trough EPS would be misleading.

### Company Classifier (Code, no LLM)
Extracts ROE level + volatility, revenue growth acceleration, and CAPEX/CFO intensity from financial data. Outputs blended company-type weights (e.g., `{"cyclical": 0.5, "growth": 0.3, "theme": 0.2}`) used by the Valuation Agent to select appropriate valuation frameworks.

### Valuation Agent (LLM)
Receives blended weights from the Classifier. Outputs a PE discount chain (Industry PE 40x → cyclical discount -30% → ROE discount -20% → CAPEX discount -15% → growth premium +20% = final 22x) with structured JSON, plus multi-framework valuation ranges (conservative PE-normalized, blended, optimistic PEG).

### Critics — Three-Way Parallel Review (LLM × 3 + Code)
- **Logic Critic**: Checks causal chain coherence, over-confident language, rating-action consistency
- **Financial Critic**: Code-layer pre-check (FCF = CFO − CAPEX, PE/ROE mismatch, CFO/depreciation distortion) → feeds findings into LLM for semantic review of financial data interpretation
- **Industry Critic**: Verifies competitive claims, technology positioning, supply chain analysis

Outputs merged and deduplicated into a structured fix list via a code-layer Aggregator.

### Repair Agent (LLM)
Receives the structured fix list from Critics. Auto-corrects the analysis JSON: tones down absolute claims, fixes data misinterpretations, adds missing risks. Only modifies flagged fields — does not regenerate the entire report.

### Reporter Agent (LLM)
Generates the final narrative, constrained by both Critic findings and Repair corrections.

### Audit Engine (Code, no LLM)
9 deterministic checks + FCF warning injection. When the scoring engine's conservative PE anchor conflicts with the LLM's growth narrative, a `[Framework Divergence]` section presents both perspectives rather than forcing one to win.

---

## Execution Trace

Every pipeline run produces an observable trace:

```
📡 Data — SUCCESS | 预取1只股票
  Latency: 4800ms | Actions: ✅财报 ✅估值 ✅行情 ✅行业 ✅评分 ✅公告

🏷️ Classify — SUCCESS | 50%周期制造 + 30%成长 + 20%主题
  Metrics: ROE 0.056, volatility 0.8, CAPEX/CFO 1.4

🧠 Analysis — SUCCESS | 投资分析生成完成
  Output: 8500 chars | RAG: 行业模板(半导体): 2条

📊 Valuation — SUCCESS | PE折价链: 40→周期-30%→ROE-20%→CAPEX-15%→成长+20%=22x
  Frameworks: PE(正常化利润), EV/EBITDA, PEG

🔍 Critics — WARNING | 三路审查: Logic:2 Financial:3 Industry:1
  Code findings: FCF=-17亿, PE/ROE=17x mismatch

🔧 Repair — SUCCESS | 已修正5项问题

📝 Report — SUCCESS | 报告生成完成
  Output: 14200 chars | Audit: 0/3 retries | Precheck: 通过
```

This trace is visible in the Streamlit UI as a collapsible panel, making the system's decision process auditable.

---

## Engineering Highlights

### Deterministic-Probabilistic Boundary
Financial calculations (EPS, PE, ROE, scores, fair value, dilution ratios) are pure Python functions. The LLM never touches numbers it can get wrong. This guarantees: same stock → same scores every time (verified: 100-run consistency >96%).

### Structured Output / Guardrails
LLM nodes (Analyst, Valuation, Critic, Repair, Audit) use `with_structured_output` with Pydantic schemas (`AnalystOutput`, `ValuationOutput`, `CriticOutput`, `AuditOutput`). If framework-level structured decoding fails, a multi-slot fallback chain falls back to raw text parsing with schema validation. This eliminates the fragility of `json.loads` + regex.

### LLM Fallback Chain
Three configurable LLM slots (slot 1 required, slot 2/3 optional) provide automatic failover. If the primary model fails or does not support structured output, the chain tries the next configured slot. All failures are logged; complete failure raises a clear `RuntimeError`.

### Dual-Layer Verification
- **Code layer (Audit Engine)**: 8 pre-checks on mathematical consistency, unit validity, field completeness
- **LLM layer (Critic Agent)**: Semantic review of logic, assumptions, and narrative quality
- **Code pre-check**: If all deterministic checks pass, the LLM auditor runs in "warning-only" mode, avoiding unnecessary retries

### Conditional Routing
LangGraph conditional edges route the workflow based on state: missing data triggers a re-fetch, and Critics with no issues skip the Repair node. This makes error recovery part of the graph rather than ad-hoc exception handling.

### RAG Knowledge System
8 industry templates covering semiconductors, power/energy, pharma, consumer, optical modules/comms, manufacturing/new energy, financials, and real estate. Each template spans: industry cycle, valuation patterns (with historical PE ranges), competitive landscape, supply chain position, and key metrics. Embedded via ONNX MiniLM (local, no API calls). Retrieved knowledge is injected into the Analyst and Valuation Agent contexts.

### Scoring Engine with Growth Premium
Fair value PE = Industry PE anchor × Financial quality multiplier × Growth premium. S-grade growth (≥9/10) gets ×1.8 PE multiplier. Cash flow quality lifts the quality floor for asset-heavy companies with strong operations but low accounting ROE (e.g., semiconductor packaging, where heavy depreciation depresses ROE but operating cash flow is 6× net profit).

### Framework Divergence Handling
When the conservative scoring engine (PE-based) conflicts with the growth narrative (PEG-based), the system does NOT force consistency. Instead, it presents both frameworks side-by-side with concrete entry prices, position sizing, and a decision framework — letting the user choose their investment philosophy.

### Data Source Tier System
Three configurable tiers (FREE/PREMIUM/INSTITUTIONAL) with pluggable premium data slots (management profiles, institutional holdings, supply chain, ESG). When premium sources are unavailable, the system gracefully degrades with explicit "data not available" markers rather than hallucinating.

---

## Features

**Data Layer**
- Financial statements (8 quarters, parent + deducted profit)
- Valuation metrics (ROE, gross/net margin, EPS, BPS, total shares)
- Real-time stock prices and K-line charts (intraday/5-day/daily/weekly/monthly)
- Announcement scanning (20 items, 3-level priority, dilution detection with auto-correction, earnings flash report content extraction)
- Sector fund flow and market breadth monitoring
- Sector momentum scoring (fund flow + relative strength + concentration, 4-tier temperature gauge)

**Analysis Layer**
- 10-step investment framework (company → moat → cycle → financials → growth → expectations → scenarios → catalysts → decision)
- 6-dimension scoring card (profitability, growth, financial health, valuation, industry outlook, market recognition)
- Scenario valuation (pessimistic/base/optimistic with probability weighting)
- Catalyst tracking and falsification conditions
- Long-term structural review (industry endgame, management, ultimate risks)

**Agent Layer**
- Intent routing (Chat / Deep Analysis / Phantom Hunter)
- Structured LLM outputs: Analyst, Valuation, Critic, Repair, Audit use Pydantic schemas
- Critic Agent with structured findings (logic flaws, over-optimism, missing risks)
- Valuation Agent with company-stage classification
- Audit Engine with 9 deterministic checks + code pre-check
- LangGraph conditional edges: data missing → re-fetch, Critic clean → skip Repair
- 4-level escalation (warning → surgical fix → analyst retry → circuit breaker)
- Processing marker system to prevent double-processing across retries
- 3-slot LLM fallback chain (slot 1 required, slot 2/3 optional)

**Output Layer**
- Structured markdown report with scoring card, valuation chain, and audit summary
- Framework divergence A/B comparison (value vs. trend, with price-stop vs. logic-stop)
- Execution trace with per-phase status, latency, and findings
- Tool call + RAG query evidence footer

**Evaluation**
- Built-in evaluation page: N stocks × M runs → score consistency, field completeness, tool success rate

---

## Demo

**Example: 长电科技 (600584) — Semiconductor Packaging**

Pipeline: Data → Analysis → Valuation → Critic → Report

Key outputs:
- Identified as "cyclical recovery with AI growth overlay"
- Conservative fair value: 25.85元 (TTM EPS 0.92 × PE 40 × quality 0.7)
- Forward sensitivity: if EPS recovers to 2.76, fair value → 77.28元
- Critic found 3 logic flaws, 2 over-optimistic claims, 2 missing risks
- Framework divergence presented: value framework (≤17元 buy zone) vs. trend framework (≤55元 buy zone)

**Example: 大唐发电 (601991) — Thermal Power**

Pipeline detected dilution event (定增 25.9亿股, 12.6% dilution) → auto-adjusted fair value, scenario prices, and forward PE by coefficient 0.874 → all dependent fields recalculated. Audit passed all 8 checks.

---

## Roadmap

**Current (v0.3)**
- [x] Multi-agent workflow (Data → Analyst → Valuation → Critic → Reporter → Audit)
- [x] RAG knowledge augmentation (8 industry templates)
- [x] Dual-layer verification (code + LLM)
- [x] Framework divergence handling
- [x] Execution trace visualization
- [x] Evaluation benchmark page
- [x] Sector momentum scoring
- [x] Structured output with Pydantic schemas for all LLM nodes
- [x] LangGraph conditional edges for data retry and Critic → Repair skip
- [x] 3-slot LLM fallback chain

**Next**
- [ ] Backtesting engine: max drawdown, Sharpe, win rate, daily PnL curve
- [ ] Agent evaluation benchmark suite: N stocks × M runs → return distribution
- [ ] Premium data source integration (Wind/Choice API for management + institutional data)
- [ ] Human-in-the-Loop: `interrupt()` before simulated order execution
- [ ] Streaming: `graph.stream()` / `astream_events()` with live frontend trace
- [ ] Agent trace export (LangSmith-compatible format)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph StateGraph + LangChain tools |
| LLM | DeepSeek / OpenAI / Anthropic (3-slot configurable fallback chain) |
| State Persistence | LangGraph SqliteSaver |
| Vector Database | ChromaDB + ONNX MiniLM-L6-V2 (local embedding) |
| Data Sources | Sina Finance + EastMoney DataCenter + THS 10jqka |
| Frontend | Streamlit + Plotly |
| Cache | Local memory (TTL) / Redis (configurable) |
| Data Tier | FREE / PREMIUM / INSTITUTIONAL (pluggable premium slots) |
| Testing | 68 e2e tests (compilation, data tools, scoring consistency, output compliance, report quality guards, structured output, graph routing, LLM fallback) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
git clone <repo-url>
cd finbrain

pip install python-dotenv langgraph langchain langchain-openai chromadb
pip install pdfplumber python-docx plotly streamlit py-mini-racer
```

### Configuration

Create `configs/.env`:

```env
# LLM (3-slot fallback chain: slot 1 required, slot 2/3 optional)
LLM_SLOT_1_PROVIDER=deepseek
LLM_SLOT_1_MODEL=deepseek-chat
LLM_SLOT_1_API_KEY=<your-api-key>
LLM_SLOT_1_BASE_URL=https://api.deepseek.com

# LLM_SLOT_2_PROVIDER=openai
# LLM_SLOT_2_MODEL=gpt-4o
# LLM_SLOT_2_API_KEY=<your-openai-key>

# LLM_SLOT_3_PROVIDER=anthropic
# LLM_SLOT_3_MODEL=claude-sonnet-5
# LLM_SLOT_3_API_KEY=<your-anthropic-key>

# Data tier (FREE | PREMIUM | INSTITUTIONAL)
FINBRAIN_DATA_TIER=FREE

# Optional: Redis cache
# REDIS_URL=redis://localhost:6379
```

Old single-provider format (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `DEEPSEEK_API_KEY`, ...) is still read for backward compatibility, but the Settings UI saves in the new 3-slot format.

### Run

```bash
# CLI
python run.py

# Web UI
streamlit run frontend/app.py
```

Type a stock code or name (e.g., "分析长电科技600584") to generate a full research report. Use the Market page for sector momentum, the Evaluation page to benchmark agent reliability.

---

## Project Structure

```
finbrain/
├── run.py                      # CLI entry point
├── configs/
│   ├── .env                    # LLM + data source configuration
│   ├── strategies.json         # 3 strategy presets
│   └── scoring.json            # Valuation weights, industry PE anchors, safety margins
├── backend/
│   ├── agent.py                # StateGraph, all agent nodes, prompts (~2900 lines)
│   ├── schemas.py              # Pydantic schemas for structured LLM output
│   ├── tools.py                # 14 data tools + scoring + formatting (~1700 lines)
│   ├── scoring.py              # Deterministic scoring engine with growth premium
│   ├── scoring_config.py       # Typed config loader
│   ├── accounting_rag.py       # 4-KB RAG system (accounting/industry/trading/youzi)
│   ├── rag.py                  # Youzi (游资) knowledge base
│   ├── datasource_tier.py      # Pluggable data source tier system
│   ├── evaluation.py           # Agent evaluation engine (N stocks × M runs)
│   ├── cache.py                # Dual-mode cache (local memory / Redis)
│   ├── api.py                  # FastAPI data service (remote mode)
│   ├── client.py               # Client-side LLM wrapper (remote mode)
│   ├── scheduler.py            # Scheduled data pre-fetching
│   ├── stock_map.py            # 5000+ stock name-to-code mappings
│   └── portfolio.py            # Mock trading portfolio
├── frontend/
│   ├── app.py                  # Streamlit UI (7 pages)
│   └── kline_chart.py          # K-line chart module
├── tests/
│   └── test_e2e.py             # 54 e2e tests
├── docs/                       # Documentation
└── data/
    ├── uploads/                # User-uploaded documents
    └── raw/chroma/             # ChromaDB vector store
```

---

## License

MIT
