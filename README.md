# Parking Reservation Chatbot

[![CI](https://github.com/VladKovalskiy/parking-reservation-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/VladKovalskiy/parking-reservation-chatbot/actions/workflows/ci.yml)

A RAG chatbot for parking-space information and reservation, with a
human-in-the-loop confirmation step before any booking is made. University
course project, delivered in 4 stages.

| Stage | Content | Status |
|-------|---------|--------|
| 0 | Environment, CI, project skeleton | ✅ |
| 1 | RAG pipeline, vector store, guardrails, evaluation | 🚧 |
| 2 | — | ⬜ |
| 3 | — | ⬜ |
| 4 | — | ⬜ |

Each stage is submitted as a git tag `stage-N` (`stage-0`, `stage-1`, …),
tagging the exact commit graded for that stage. The course submission form
gets a link to that tag, e.g.:

```
https://github.com/VladKovalskiy/parking-reservation-chatbot/tree/stage-1
```

Tag a stage once it's done and pushed:

```bash
git tag stage-1
git push origin stage-1
```

## Stack and why

| Component | Choice | Rationale |
|-----------|--------|-----------|
| LLM | Anthropic Claude | Haiku for cheap internal steps (routing, extraction, guardrails), Sonnet for generation |
| Embeddings | `sentence-transformers` (multilingual-e5) | Anthropic has no embeddings API; a local model runs offline, is free, and lets the evaluation compare models |
| Vector store | Milvus (Lite in dev/CI, standalone in demo) | one `langchain-milvus` library for both modes — local dev needs no Docker, the demo gets a full server |
| Dynamic data | PostgreSQL | availability, hours, and prices change on their own schedule — not a job for vector search (see [`docs/sql-schema.md`](docs/sql-schema.md)) |
| Guardrails | Presidio | pre-trained NLP models for PII detection |
| Orchestration | LangGraph | needed for stateful dialogue and human-in-the-loop confirmation in later stages |

## Setup

Prerequisites:

- [`uv`](https://docs.astral.sh/uv/) — installs and manages Python for you:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (`uv` will install Python 3.12 itself on first use — no separate Python setup needed.)
- Docker + Docker Compose — **optional**, only needed for the standalone
  Milvus demo mode. Local development and tests use Milvus Lite (a local
  file) and don't need Docker at all.
- If you're on Windows, work inside WSL2 and keep the repo on the Linux
  filesystem (e.g. `~/dev/parking-bot`), **not** under `/mnt/c` — I/O on a
  mounted Windows drive is slow and breaks file watchers.

Steps:

```bash
# 1. Install dependencies (downloads Python 3.12 automatically if needed)
make install

# 2. Configure environment
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY to a real key
# (not required for unit tests, only for the LLM smoke test and the chatbot itself)

# 3. Install pre-commit hooks
make hooks

# 4. Verify the setup
make test
```

If `make test` passes, the setup is complete — unit tests need no API key
and no running services (see [Testing](#testing)).

### Optional: Milvus standalone + PostgreSQL (demo mode)

```bash
make up      # starts Milvus standalone, Postgres, and Attu (UI) on http://localhost:8000
```

Then switch `.env` from Milvus Lite to standalone by setting:

```
MILVUS_URI=http://localhost:19530
```

(Leave `MILVUS_LITE_PATH` as-is — it's only used when `MILVUS_URI` is unset.
See [Known traps](CLAUDE.md#known-traps) in `CLAUDE.md` for why these are two
separate variables.) Stop the stack with `make down`.

## Usage

The project is at stage 1 (RAG pipeline) — there is no running chat
interface yet (`api/` and `graph/` are scaffolding for later stages). What
you can run today:

- **Manual smoke checks** (`scripts/smoke_*.py`) — these hit live services
  (cost tokens, download a model), so they are not pytest tests and never
  run in CI:
  ```bash
  # Verifies ANTHROPIC_API_KEY reaches both configured models
  uv run python scripts/smoke_anthropic.py

  # Loads the embedding model, indexes 3 documents into Milvus, checks retrieval
  uv run python scripts/smoke_embeddings.py

  # Ingests data/static/, asks a grounded question and an out-of-scope one
  uv run python scripts/smoke_rag_chain.py
  ```
  `smoke_embeddings.py` downloads the `intfloat/multilingual-e5-base` model
  (~1 GB) on first run, and connects to whichever Milvus target is
  configured in `.env` — Lite by default, or standalone if `MILVUS_URI` is
  set — with no code changes needed to switch between them.
- **Unit tests** — see [Testing](#testing) below.

## Structure

```
src/parking_bot/
├── config.py       # typed config; every provider swappable via env
├── ingestion/      # document loading and chunking
├── retrieval/      # vector store, retriever
├── llm/            # chat + embeddings factories, plus fake backends for tests
├── rag/            # grounded RAG chain + static/dynamic SQL-vs-RAG router
├── guardrails/     # PII filtering
├── eval/           # Recall@K / Precision@K harness (make eval)
├── db/             # SQLAlchemy models + init/seed for dynamic data (docs/sql-schema.md)
├── booking/        # interactive booking-field intake: validate, ask, persist a draft
├── graph/          # LangGraph state (stage 2+)
└── api/            # interface (stage 2+)
data/
├── static/         # documents for the vector store: general info, location,
│                   # booking process, rules — see the static/dynamic split
│                   # in docs/sql-schema.md
└── eval/           # golden_set.jsonl — question → relevant-document pairs
docs/
├── sql-schema.md   # PostgreSQL schema design for dynamic data
└── evaluation.md   # Recall@K/Precision@K/latency report, embedding model comparison
scripts/            # manual smoke checks (not run in CI)
tests/              # pytest unit tests (offline, no live services)
```

## Testing

Unit tests need neither API keys nor running services: `tests/conftest.py`
forces `EMBEDDING_PROVIDER=fake` and an in-memory-style Milvus Lite store
(`MILVUS_LITE_PATH=:memory:`). Tests that require a real Milvus/Postgres or
API access are marked `integration` and excluded from CI.

```bash
make test       # unit tests only (excludes the integration marker) — what CI runs
make test-all   # everything, including integration tests
make lint       # ruff check + format --check
make fmt        # ruff format + check --fix — run before committing
```

`mypy` also runs in CI (`uv run mypy src`) but is currently non-blocking.

## Evaluation

`data/eval/golden_set.jsonl` holds 28 hand-labeled question → relevant-document
pairs against `data/static/`, written before any retrieval code so the set
isn't fitted to what the system already does.

```bash
make eval
```

Re-ingests `data/static/` (using whichever embedding provider/model is
configured in `.env`), runs every golden-set question through retrieval, and
computes Recall@K and Precision@K at several cutoffs (default `k=1,3,5,10`).
Writes `data/eval/report.json`, e.g.:

```json
{
  "num_questions": 28,
  "metrics": {
    "1": {"recall_at_k": 0.84, "precision_at_k": 0.86},
    "5": {"recall_at_k": 0.96, "precision_at_k": 0.20}
  },
  "embedding_provider": "local",
  "embedding_model": "intfloat/multilingual-e5-base"
}
```

Because embeddings/vector-store choice comes entirely from config (ADR-001/
002/003), re-running the report against a different embedding model is a
`.env` change, not a code change — this is what makes model comparison cheap.
Pass `--k`, `--golden-set`, `--report`, or `--skip-ingest` to
`uv run python -m parking_bot.eval.harness` to override the defaults.

See [`docs/evaluation.md`](docs/evaluation.md) for the full report: methodology,
Recall@K/Precision@K and latency tables, and a `multilingual-e5-base` vs.
`multilingual-e5-small` comparison that justifies ADR-001's choice of local,
swappable embeddings.

## Dynamic data (PostgreSQL)

Spaces, tariffs, operating hours, and reservations are not a RAG problem
(see [Stack](#stack-and-why)). The schema design, the static/dynamic
boundary, and the reasoning behind key decisions (double-booking guard,
price snapshotting, availability as a query rather than a table) are in
[`docs/sql-schema.md`](docs/sql-schema.md).

```bash
make up        # starts Postgres (+ Milvus standalone + Attu)
make db-init   # creates the schema — src/parking_bot/db/init_db.py
make db-seed   # init + loads demo spaces/tariffs/hours/a reservation
```

SQLAlchemy models live in [`src/parking_bot/db/models.py`](src/parking_bot/db/models.py),
one per `docs/sql-schema.md` table. There is deliberately no `Availability`
model — [`db/availability.py`](src/parking_bot/db/availability.py) answers
"which spaces are free in this window" as a query instead, matching the
design doc's reasoning. The `reservations` double-booking guard
(`EXCLUDE USING gist`) is PostgreSQL-only and can't be expressed as a
portable SQLAlchemy constraint, so it's attached via a dialect-scoped DDL
event — real on Postgres, silently absent on the SQLite used by
`tests/test_db.py`'s offline CRUD tests. That one guard is covered
separately by `tests/test_db_integration.py` (`@pytest.mark.integration`,
needs `make up`).

### Static/dynamic routing

[`rag/router.py`](src/parking_bot/rag/router.py) is what actually enforces
the static/dynamic split at answer time: `classify_question()` routes
availability/price/hours-shaped questions to a SQL query (formatted
directly into the answer — no LLM paraphrase of a price or a closing time,
by design) and routes everything else to the grounded RAG chain. Routing is
deterministic keyword matching, not an LLM call — see the module docstring
for why that's consistent with how `guardrails/pii.py` and
`booking/collector.py` made the same call for their own "cheap internal
step."

## CI

GitHub Actions runs on every push and PR: ruff (lint + format), mypy, and
pytest with coverage. No secrets are required.
