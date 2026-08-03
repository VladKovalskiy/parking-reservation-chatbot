# CLAUDE.md

Guidance for AI agents working in this repository. Read this before making changes.

## What this project is

A RAG chatbot for parking-space information and reservation, with a
human-in-the-loop confirmation step. University course project, delivered in
stages. Each stage is submitted as a git tag `stage-N` (stage-0, stage-1, …),
so the reviewer sees the exact state being graded.

## Locked decisions — do not revisit without being asked

These were deliberated and settled. Do not propose alternatives unless the user
explicitly reopens the question.

- **LLM: `claude-sonnet-4-6` only.** Sonnet 5 was rejected because it removes
  sampling parameters (setting `temperature` returns HTTP 400), and this RAG
  pipeline relies on `temperature=0` for deterministic answers. Do not suggest
  Sonnet 5 or other providers. (ADR-004)
- **Fast model: `claude-haiku-4-5`** for cheap internal steps (routing,
  extraction, guardrails). This is a separate axis from the generation model,
  not a competitor to Sonnet 4.6.
- **Embeddings: local `sentence-transformers` (`intfloat/multilingual-e5-base`).**
  Anthropic has no embeddings API. Local models keep CI key-free and let the
  stage-1 evaluation compare models for free. Do not suggest OpenAI/Anthropic
  embeddings. (ADR-001)
- **Vector store: Milvus.** Milvus Lite (a local `.db` file) in development and
  CI; Milvus standalone via `docker/compose.yml` for the demo. Same
  `langchain-milvus` library for both. Switched via config: `MILVUS_LITE_PATH`
  holds the local file (default), and setting `MILVUS_URI` (a real
  `http://host:port`) overrides it for standalone. These are deliberately two
  separate env vars, not one — `pymilvus` itself reads `MILVUS_URI` for its own
  global default connector at import time and requires a network address, so a
  local path can never live under that name. (ADR-002)
- **Dynamic data (availability, hours, prices, reservations): PostgreSQL.**
  Static data (general info, location, booking process) goes in the vector
  store. Keep this split.
- **Guardrails: Microsoft Presidio** for PII detection.
- **Config: everything via environment** (`pydantic-settings`, `.env`). No
  hardcoded provider choices — this is what makes the evaluation report cheap.
  (ADR-003)
- **Product language: English.** Bot conversation, static parking documents,
  and the golden set are all English. This simplifies Presidio (built-in
  English recognizers instead of custom Ukrainian ones). Code, comments,
  docstrings, and commits are English too. (ADR-005)

## Environment

- Windows + WSL2. The project lives in the WSL filesystem (`~/dev/parking-bot`),
  **never** under `/mnt/c` — mounted-drive I/O is slow and breaks file watchers.
- Python 3.12, managed by `uv`. Use `uv run …` and `uv sync`, not bare `pip`.
- `uv.lock` is committed and authoritative; regenerate it deliberately, not by
  accident.

## Common commands

```bash
make install    # uv sync --extra dev
make test       # pytest, excluding the `integration` marker
make test-all   # all tests
make lint       # ruff check + format --check
make fmt        # ruff format + check --fix   (run before committing)
make up / down  # docker compose: Milvus standalone + Postgres + Attu
make eval       # re-ingest data/static/, score Recall@K/Precision@K on the golden set
make db-init    # create the Postgres schema (docs/sql-schema.md)
make db-seed    # init + load demo spaces/tariffs/hours/reservation
```

## Conventions

- **Commits: Conventional Commits.** `feat:`, `fix:`, `chore:`, `docs:`,
  `style:`, `refactor:`, `test:`. Imperative subject; body explains *why*.
  Keep commits atomic — one logical change each; never mix a formatting sweep
  with a behavior change.
- **Tests must run offline.** `tests/conftest.py` forces `EMBEDDING_PROVIDER=fake`
  and an in-memory store so unit tests need no API key and no running services.
  Tests that require real Milvus/Postgres/API access get the `integration`
  marker and are excluded from CI. Preserve this — never write a unit test that
  needs a live key.
- **Smoke scripts vs tests.** `scripts/smoke_*.py` are manual checks that hit
  live services (cost tokens, need keys). They are not pytest tests and must
  not run in CI.
- Run `make fmt` and `make test` before every commit. CI runs the same ruff +
  pytest steps, so green locally means green in CI.

## Known traps

- **`multilingual-e5` requires `query:` / `passage:` prefixes** when encoding.
  Queries get `query: `, documents get `passage: `. Omitting them noticeably
  degrades retrieval. If the embedding model is ever switched to a non-e5 model
  (e.g. `bge-base-en`), remove these prefixes — they are e5-specific.
- **`temperature` and Sonnet 4.6:** `temperature=0` works and is wanted. Do not
  add sampling parameters beyond what config exposes.
- **Never put a local path in an env var literally named `MILVUS_URI`.**
  `pymilvus` calls `load_dotenv()` itself and reads `MILVUS_URI` for its own
  global default connector at import time — before any of our code runs — and
  requires a network address there. A local Milvus Lite path in that exact env
  var crashes `import pymilvus` outright. Local dev/CI path goes in
  `MILVUS_LITE_PATH` instead; see ADR-002.
- Files copied from Windows into WSL can carry `:Zone.Identifier` sidecar files
  and `777` permissions. Prefer creating/editing files directly in WSL.
- **Presidio's default `AnalyzerEngine()` silently downloads `en_core_web_lg`
  (~400 MB)** the first time it's used, because its bundled `conf/default.yaml`
  hardcodes that model — regardless of what spaCy models are already
  installed. `guardrails/pii.py` avoids this by building the NLP engine
  explicitly (`NlpEngineProvider()` then overriding `.nlp_configuration["models"]`)
  to point at the pinned, much smaller `en_core_web_sm` (see
  `pyproject.toml`'s `en-core-web-sm` direct-URL dependency, which needed
  `tool.hatch.metadata.allow-direct-references = true` to install). Building
  the NLP config from scratch instead of overriding just `models` loses
  `default.yaml`'s `labels_to_ignore` filter and causes false-positive PII
  matches (e.g. "Reservations" tagged as `ORGANIZATION`) — always start from
  the bundled default and only override the model.
- **Presidio's `UkVehicleRegistrationRecognizer` (used for car-plate
  detection) ships `enabled: false`** in Presidio's own default recognizer
  list — it's opt-in even when using the `countries=["uk"]` filter on
  `load_predefined_recognizers()`, since that filter only ever narrows
  recognizers that are already enabled. Add it explicitly via
  `registry.add_recognizer(UkVehicleRegistrationRecognizer())` instead.
- **Presidio's `PhoneRecognizer` has a fixed base score of 0.4**, boosted to
  ~0.75 only when a context word (e.g. "phone", "number") appears near the
  match — a bare number with no such context stays at 0.4 and won't clear
  `pii_score_threshold`'s default of 0.5. This is inherent to Presidio, not a
  bug to "fix"; write PII test fixtures with realistic surrounding context.
- **`db/models.py` must be imported before `Base.metadata.create_all()`, or
  it silently creates zero tables.** Declarative model classes only register
  themselves on `Base.metadata` as a side effect of the module executing —
  `db/init_db.py` imports `parking_bot.db.models` purely for that side
  effect (`# noqa: F401`), not because it uses any name from it. Confirmed
  live against Postgres: dropping that import made `init_db()` "succeed"
  (no exception) while creating no tables at all. `db/seed.py` happens to
  import `models` anyway (it uses `Space`, `Tariff`, ...), which is why this
  only broke `init_db.py` run standalone.
- **`sqlalchemy.dialects.postgresql.ExcludeConstraint` can't live in a
  model's `__table_args__`** if that model also needs to be created on
  SQLite (as `tests/test_db.py` does) — compiling it against the SQLite
  dialect raises `UnsupportedCompilationError`, aborting `create_all()`
  entirely. `db/models.py` instead attaches the `reservations` double-
  booking guard as a raw-SQL `DDL(...)` on an `after_create` event scoped
  with `.execute_if(dialect="postgresql")`, so SQLite silently skips it and
  Postgres gets the real `EXCLUDE USING gist` constraint (verified against a
  live `make up` Postgres — see `tests/test_db_integration.py`).

## Structure

```
src/parking_bot/
├── config.py       typed config; every provider swappable via env
├── ingestion/      document loading and chunking
├── retrieval/      vector store, retriever
├── llm/            chat + embeddings factories (+ fake backends for tests)
├── rag/            grounded RAG chain: prompt + retrieval -> generation
├── guardrails/     PII filtering
├── eval/           Recall@K / Precision@K harness (make eval)
├── db/             SQLAlchemy models + init/seed for dynamic data (docs/sql-schema.md)
├── booking/        interactive booking-field intake: validate, ask, persist a draft
├── graph/          LangGraph state (stage 2+)
└── api/            interface
data/
├── static/         documents for the vector store
└── eval/           golden set for Recall@K / Precision
scripts/            manual smoke checks (not CI)
```

## Documentation

- `README.md` in the repo is the source of truth for setup/usage.
- A Notion page ("Parking Reservation Chatbot") holds the decision journal:
  stack rationale, ADR-001..005, stage plan, and a task tracker. When a
  significant architectural decision is made, it belongs there as a new ADR.
