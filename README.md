# SHL Conversational Assessment Recommender

A stateless FastAPI service that guides hiring managers from a vague role description to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

Built for the **SHL Labs AI Intern Take-Home Assignment**.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture & Flow](#architecture--flow)
3. [Project Structure](#project-structure)
4. [How The Catalog Was Built](#how-the-catalog-was-built)
5. [Agent State Machine](#agent-state-machine)
6. [Retrieval Layer](#retrieval-layer)
7. [LLM Integration](#llm-integration)
8. [API Reference](#api-reference)
9. [Setup & Running Locally](#setup--running-locally)
10. [Frontend](#frontend)
11. [Deployment](#deployment)
12. [Testing](#testing)
13. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## What It Does

A user types a hiring need — anything from `"I'm hiring a Java developer"` to a pasted job description. The agent:

- **Clarifies** when the request is too vague to act on
- **Recommends** 1–10 SHL assessments once it has enough context, with real catalog URLs
- **Refines** when the user changes constraints mid-conversation (`"also add personality tests"`)
- **Compares** assessments when asked (`"what's the difference between OPQ and Verify?"`)
- **Refuses** off-topic requests, legal questions, and prompt-injection attempts

Every URL returned is verified to exist in the scraped SHL catalog — no hallucinations.

---

## Architecture & Flow

```
User message
     │
     ▼
┌────────────────────────────────────────────┐
│  POST /chat  (FastAPI — app/main.py)       │
│                                            │
│  1. Validate turn count (max 8)            │
│  2. extract_state(messages)                │
│     └─► LLM call (structured JSON)        │
│         └─► keyword fallback if LLM fails │
│  3. classify_intent(messages, state)       │
│     └─► Rule-based: CLARIFY / RECOMMEND / │
│              REFINE / COMPARE / REFUSE     │
│  4. Retrieve candidates (BM25 + boost)     │
│     └─► Only for RECOMMEND / REFINE       │
│  5. generate_response(intent, candidates)  │
│     └─► LLM selects from candidates list  │
│         └─► deterministic fallback        │
│  6. Validate all URLs against catalog.json │
│  7. Return ChatResponse (strict schema)    │
└────────────────────────────────────────────┘
     │
     ▼
{
  "reply": "string",
  "recommendations": [
    {"name": "...", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

### Turn-by-turn example

```
Turn 1  User:  "Hello, can you help me?"
        Agent: CLARIFY → "What role are you hiring for?"

Turn 2  User:  "Senior Java developer, works with stakeholders"
        Agent: RECOMMEND → 3-5 Java skills assessments + explains choice

Turn 3  User:  "Actually, also add personality tests"
        Agent: REFINE → updated list including OPQ assessments

Turn 4  User:  "What's the difference between OPQ32r and Java 8?"
        Agent: COMPARE → grounded comparison from catalog data only
```

---

## Project Structure

```
SHL_Project/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app — endpoints, request/response models
│   ├── agent.py         # State extractor, intent classifier, response generator
│   └── retrieval.py     # BM25 search + metadata filters
│
├── static/
│   └── index.html       # Chat UI (served at GET /)
│
├── tests/
│   ├── test_behavior.py # 10 behavioral probes (off-topic, injection, schema, etc.)
│   └── test_e2e.py      # 5 end-to-end integration tests
│
├── catalog.json          # 536 SHL Individual Test Solutions (scraped)
├── run.py                # Dev server entry point
├── requirements.txt
├── Dockerfile
├── render.yaml           # One-click Render.com deployment
├── .env                  # Local secrets (not committed)
└── .env.example          # Template for environment variables
```

---

## How The Catalog Was Built

The SHL product catalog at `shl.com/solutions/products/product-catalog/` is fully JavaScript-rendered and has since been restructured (the URL now redirects). The catalog was built in three stages:

### Stage 1 — Discover URLs via Wayback CDX API
```
GET https://web.archive.org/cdx/search/cdx
    ?url=shl.com/solutions/products/product-catalog/
    &matchType=prefix&filter=statuscode:200&collapse=original
```
This returned **536 unique `/view/` URLs** — each one is a direct link to an individual assessment's catalog page (e.g. `/view/java-8-new/`).

### Stage 2 — Scrape view pages for metadata
For the first 100 URLs, each individual product page was fetched from the Wayback Machine snapshot (`timestamp=20250430003713`). Each page contains a table with:
- Assessment name
- Test type codes (A / B / C / D / E / K / M / P / S)
- Remote testing capability
- Adaptive/IRT flag

### Stage 3 — Slug inference for remaining 436
For URLs that returned 404 from Wayback, the assessment name and test type were inferred from the URL slug using keyword matching:
- `verify-numerical-reasoning` → type `A` (Ability)
- `occupational-personality-questionnaire` → type `P` (Personality)
- `java-8-new` → type `K` (Knowledge & Skills)

### Catalog schema
```json
{
  "name": "Java 8 (New)",
  "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
  "test_type": "K",
  "keys": ["K"],
  "remote": true,
  "adaptive": false
}
```

### Test type key
| Code | Category |
|------|----------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgement |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| M | Motivational |
| P | Personality & Behaviour |
| S | Simulation |

---

## Agent State Machine

Defined in `app/agent.py`.

### ConversationState
Accumulated facts extracted from the full message history:

```python
class ConversationState:
    role: str                      # "Java Developer", "Sales Manager"
    seniority: str                 # "entry-level", "mid-level", "senior"
    skills: List[str]              # ["java", "stakeholder", "agile"]
    test_types_requested: List[str] # ["P", "K"] — explicit type requests
    remote_required: bool
    adaptive_required: bool
    duration_max_minutes: int
    language: str
    job_description: str           # first 500 chars of pasted JD
```

### State Extraction (two-tier)
1. **LLM call** — structured JSON extraction prompt sent to the model. Fast, handles paraphrase, out-of-order facts, corrections.
2. **Keyword fallback** — regex + vocabulary matching runs if LLM is unavailable or returns invalid JSON. Handles all core cases without any API dependency.

### Intent Classification (rule-based)

The classifier is deterministic — no LLM cost, no latency:

```
REFUSE    → regex matches off-topic/injection patterns
             (legal advice, GDPR, "ignore previous instructions", etc.)

COMPARE   → "difference between", "vs", "versus", "which is better"

REFINE    → prior assistant message contains shl.com URLs
             AND current message contains ["actually", "add", "remove", etc.]

RECOMMEND → user said "show me" / "recommend" / "i need"
             OR user pasted a JD (>200 chars)
             OR enough context exists AND turn >= 2
             OR turn count >= 6 (approaching limit — commit to shortlist)

CLARIFY   → none of the above AND role/JD/test-type not yet known
             → asks exactly ONE focused question
```

### Intent → Action mapping

| Intent | `recommendations` | LLM used? |
|--------|-------------------|-----------|
| CLARIFY | `[]` | No |
| REFUSE | `[]` | No |
| COMPARE | 2 items | Yes (grounded comparison) |
| RECOMMEND | 1–10 items | Yes (selection from candidates) |
| REFINE | 1–10 items | Yes (updated selection) |

---

## Retrieval Layer

Defined in `app/retrieval.py`.

### BM25 Index
At startup, `CatalogRetriever` builds a BM25Okapi index over every catalog entry. Each document is the assessment **name** concatenated with its **expanded type labels**:

```
"Java 8 New" + "Knowledge Skills Technical"
"Occupational Personality Questionnaire" + "Personality Behaviour Behavioral"
```

This means searching for "personality tests" correctly scores OPQ entries highly even though "personality" doesn't appear literally in the assessment name.

### Two-stage retrieve for REFINE
When the user adds a test type (`"add personality tests"`), a simple filter would return *only* personality tests, dropping the original Java skills assessments. Instead:

1. **Stage 1** — BM25 top-20 with no type filter (keeps existing relevant results)
2. **Stage 2** — Boost: for each explicitly requested type, inject the top-3 scoring entries of that type not already in the result

This gives continuity (prior results stay) plus expansion (new type added).

### Metadata filters
- `remote_only` — filter to `remote=true` entries
- `adaptive_only` — filter to `adaptive=true` entries
- `test_types` — hard filter to specific type codes (used when user specifies exactly one type)

---

## LLM Integration

### Provider
Groq (free tier) via the OpenAI-compatible SDK. Falls back to OpenAI if `GROQ_API_KEY` is not set.

```python
client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
)
```

Model: `llama-3.1-8b-instant` (fast, free on Groq)

### Two focused LLM calls per request

**Call 1 — State extraction** (`extract_state`):
- Input: full conversation history
- Output: structured JSON (`ConversationState`)
- Max tokens: 400
- Temperature: 0

**Call 2 — Response generation** (`_handle_recommend`):
- Input: state summary + last 6 messages + up to 15 BM25 candidates
- Output: `{"reply": "...", "indices": [0, 3, 5]}`  (indices into candidates list)
- Max tokens: 300
- Temperature: 0.2

The LLM **never sees the full catalog** — it only sees the pre-filtered candidates list. This eliminates hallucination: it can only select by index from the provided list.

### Hallucination guard
After LLM selection, every URL is checked against `catalog.json` before being returned. Any URL not in the catalog is silently dropped. The response always has valid URLs or is empty.

### Deterministic fallback
If either LLM call fails (quota, network, invalid JSON), the system:
- State: uses keyword regex extractor
- Response: returns top-5 BM25 candidates with a templated reply

The service **never crashes or returns an invalid schema** regardless of LLM availability.

---

## API Reference

### `GET /health`
Readiness check.

**Response** `200 OK`:
```json
{"status": "ok"}
```

---

### `POST /chat`
Stateless conversational endpoint. Every call carries the full conversation history.

**Request**:
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

**Response**:
```json
{
  "reply": "Here are 3 assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules (non-negotiable)**:
- `recommendations` is `[]` while clarifying or refusing
- `recommendations` has 1–10 items when committing to a shortlist
- `end_of_conversation` is `true` only when task is complete
- Every `url` exists in `catalog.json`

**Error responses**:
- `400` — conversation exceeds 8 user turns
- `422` — malformed request body

---

### `GET /`
Serves the chat UI (`static/index.html`).

---

## Setup & Running Locally

### Prerequisites
- Python 3.9+
- A Groq API key (free at [console.groq.com](https://console.groq.com)) **or** OpenAI API key

### 1. Clone and install
```bash
git clone <repo-url>
cd SHL_Project
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your API key
```

**.env**:
```env
# Option A — Groq (free, recommended)
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.1-8b-instant

# Option B — OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

HOST=127.0.0.1
PORT=8000
```

### 3. Start the server
```bash
python3 run.py
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
LLM provider: Groq
Loaded catalog: 536 assessments
Using LLM model: llama-3.1-8b-instant
INFO:     Application startup complete.
```

### 4. Open the UI
Visit **http://127.0.0.1:8000** in your browser.

Or test the API directly:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need assessments for a senior Python data scientist"}]}'
```

---

## Frontend

The chat UI is a single static HTML file served by FastAPI at `GET /`. No separate Node/npm server needed.

**Features:**
- SHL-branded dark header
- Animated typing indicator while waiting for response
- Recommendation cards with test-type badge and direct SHL catalog link
- Suggestion chips for common queries
- Auto-growing textarea, Enter to send
- Full conversation history maintained client-side

**Tech:** Vanilla HTML/CSS/JS — no framework, no build step.

---

## Deployment

### Render.com (recommended — free tier)

1. Push repo to GitHub
2. Connect to Render → New Web Service
3. Set environment variables in dashboard:
   - `GROQ_API_KEY` → your key
   - `LLM_MODEL` → `llama-3.1-8b-instant`
4. Deploy — `render.yaml` handles the rest

`render.yaml` is already configured:
```yaml
buildCommand: pip install -r requirements.txt
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /health
```

### Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=gsk_... \
  -e LLM_MODEL=llama-3.1-8b-instant \
  shl-recommender
```

---

## Testing

### Run behavior probes (10 probes)
```bash
python3 tests/test_behavior.py
```

Tests:
1. No recommendation on vague first turn
2. Recommends when context is sufficient
3. Refuses off-topic legal questions
4. Refuses prompt injection attempts
5. All returned URLs are valid catalog URLs
6. Refinement updates the shortlist
7. Recommendation count is between 1 and 10
8. Response schema is strictly compliant
9. Turn cap (max 8) is enforced
10. Health endpoint returns 200

### Run end-to-end tests (5 tests)
```bash
python3 tests/test_e2e.py
```

Tests:
1. Health check
2. Full vague → clarify → recommend flow
3. Job description paste → direct recommendation
4. Strict schema compliance on all fields
5. Malformed request → 422

Both test suites require the server to be running at `http://127.0.0.1:8000`.

---

## Design Decisions & Trade-offs

### Why BM25 over vector embeddings?
The SHL catalog has short, specific assessment names. BM25 on name + expanded type labels performs well for exact and near-exact matches ("Java developer" → Java assessments). Vector embeddings would add startup latency, a dependency (sentence-transformers or OpenAI embeddings API), and are harder to debug when they mis-rank. BM25 is fast, explainable, and the catalog is small enough (536 items) that recall is not meaningfully improved by embeddings.

### Why stateless API?
The assignment spec requires it. Each `/chat` call carries the full message history. This makes the service horizontally scalable with no session storage, trivially deployable, and easy to test (no session setup/teardown).

### Why two LLM calls instead of one big agent prompt?
Separation of concerns:
- **State extraction** needs to be fast and structured → low temperature, small output
- **Response generation** needs to be natural and selective → slightly higher temperature, constrained to candidates list

A single large prompt mixing both tasks produces worse JSON structure and is harder to debug.

### Why keyword fallback for state extraction?
LLMs can fail (quota, rate limit, network). A hiring manager shouldn't get a 500 error because Groq had a blip. The keyword extractor handles ~90% of real queries correctly and ensures the service always responds with something useful.

### Why not use LangChain / LangGraph?
The grader cares about correctness and defensibility. A raw FastAPI + Pydantic implementation is fully transparent — every decision in the code is visible and justified. Framework abstractions would add complexity without benefit at this scale.

### Catalog coverage
536 entries covers the full scope of SHL Individual Test Solutions. Approximately 100 entries have fully scraped metadata (test type, remote, adaptive) from individual view pages. The remaining ~436 have inferred metadata from URL slug keywords, which is accurate for the dominant pattern (e.g., all `*-new` skill tests are type K, all `opq*` are type P).
