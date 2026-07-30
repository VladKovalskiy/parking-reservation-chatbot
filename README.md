# Parking Reservation Chatbot

RAG-чатбот для інформування про паркінг та бронювання місць з human-in-the-loop
підтвердженням. Навчальний проєкт, 4 стадії.

| Stage | Зміст | Статус |
|-------|-------|--------|
| 0 | Оточення, CI, скелет проєкту | ✅ |
| 1 | RAG-система, векторна БД, guardrails, evaluation | 🚧 |
| 2 | — | ⬜ |
| 3 | — | ⬜ |
| 4 | — | ⬜ |

## Стек і чому саме він

| Компонент | Вибір | Обґрунтування |
|-----------|-------|---------------|
| LLM | Anthropic Claude | доступний API; Haiku для дешевих внутрішніх кроків, Sonnet для генерації |
| Embeddings | `sentence-transformers` (multilingual-e5) | Anthropic не надає embeddings API; локальна модель працює офлайн, безкоштовно і дозволяє порівнювати моделі в evaluation |
| Vector store | Milvus (Lite у dev, standalone у demo) | одна бібліотека `langchain-milvus` для обох режимів — розробка без Docker, демо з повноцінним сервером |
| Dynamic data | PostgreSQL | наявність місць, години, ціни змінюються — це не задача для векторного пошуку |
| Guardrails | Presidio | pre-trained NLP-моделі для виявлення PII |
| Orchestration | LangGraph | потрібен для stateful-діалогу і human-in-the-loop на пізніших стадіях |

## Швидкий старт

```bash
# 1. Залежності (uv встановлює Python 3.12 автоматично)
make install

# 2. Конфіг
cp .env.example .env   # додати ANTHROPIC_API_KEY

# 3. Pre-commit хуки
make hooks

# 4. Тести
make test
```

Docker потрібен лише для демо-режиму з повноцінним Milvus:

```bash
make up      # Milvus standalone + Postgres + Attu UI на localhost:8000
# і в .env: MILVUS_URI=http://localhost:19530 (замість MILVUS_LITE_PATH)
```

## Структура

```
src/parking_bot/
├── config.py       # типізований конфіг, усі провайдери свапаються через env
├── ingestion/      # завантаження і чанкінг документів
├── retrieval/      # vector store, retriever
├── llm/            # обгортки над LLM та embeddings + fake-реалізації для тестів
├── guardrails/     # PII-фільтрація
├── graph/          # LangGraph-стани (stage 2+)
└── api/            # інтерфейс
data/
├── static/         # документи для векторної БД
└── eval/           # golden set для Recall@K / Precision
docs/
└── sql-schema.md   # дизайн PostgreSQL-схеми для динамічних даних
```

## Тестування

Юніт-тести не потребують ані API-ключів, ані запущених сервісів: `conftest.py`
примусово перемикає конфіг на fake-embeddings та in-memory сховище. Тести, що
вимагають реального Milvus чи Postgres, позначені маркером `integration` і в CI
не запускаються.

```bash
make test       # без інтеграційних
make test-all   # усі
```

## Evaluation

`data/eval/golden_set.jsonl` — розмічені пари «питання → релевантні документи».
Наповнюється паралельно з написанням документів у `data/static/`. Метрики
(Recall@K, Precision@K, latency) рахуються скриптом `make eval` — з'явиться на
stage 1.

## Dynamic data (PostgreSQL)

Місця, тарифи, години роботи і бронювання — це не задача для RAG (див.
[Стек](#стек-і-чому-саме-він)). Дизайн схеми, межа статичне/динамічне і
обґрунтування ключових рішень (double-booking guard, price snapshot,
availability як запит, а не таблиця) — у
[`docs/sql-schema.md`](docs/sql-schema.md). Реалізація (моделі, міграції)
з'явиться на пізнішому етапі stage 1.

## CI

GitHub Actions на кожен push і PR: ruff (lint + format), mypy, pytest з покриттям.
Секрети не потрібні.
