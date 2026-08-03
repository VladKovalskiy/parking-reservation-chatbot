# Retrieval evaluation report

Stage-1 evaluation of the retrieval half of the RAG pipeline against
[`data/eval/golden_set.jsonl`](../data/eval/golden_set.jsonl): 28 hand-labeled
question → relevant-document pairs, written before any retrieval code
existed so the set isn't fitted to what the system already does.

This report covers **retrieval quality and latency only** — Recall@K,
Precision@K, and query latency. It does not score generated-answer
correctness (`expected_answer_contains` in the golden set is reserved for
that; no generation-quality harness exists yet).

## Methodology

**Pipeline under test:** `data/static/*.md` → [`ingestion/loader.py`](../src/parking_bot/ingestion/loader.py)
(anchor-scoped sections) → [`ingestion/chunker.py`](../src/parking_bot/ingestion/chunker.py)
(`chunk_size=800`, `chunk_overlap=120`) → embeddings → Milvus → top-K
similarity search via [`retrieval/retriever.py`](../src/parking_bot/retrieval/retriever.py).
At these chunk sizes every one of the 4 static documents' 20 anchored
sections fits in a single chunk (20 sections in → 20 chunks indexed), so
chunking doesn't fragment any answer across chunks in this dataset.

**Metrics** ([`eval/metrics.py`](../src/parking_bot/eval/metrics.py)):

- **Recall@K** — fraction of a question's `relevant_doc_ids` that appear
  anywhere in the top-K retrieved chunks.
- **Precision@K** — fraction of the K retrieved chunks that are relevant.
  The denominator is always K (not the number of chunks actually returned),
  so precision necessarily falls as K grows given this golden set has
  exactly one relevant doc per question — that's an expected property of the
  metric on this dataset, not a retrieval defect.
- **Latency** — wall-clock time of one `retrieve()` call (query embedding +
  Milvus similarity search), timed once per question at `k = max(K)` — since
  scoring smaller K values is just slicing that same ranked list, it costs no
  extra queries.

**Environment:** Milvus Lite (local file), CPU-only inference (no usable
GPU on the dev machine this was run on) — latency numbers reflect CPU
embedding inference on that machine, not a production/GPU deployment.

**Reproduce:**

```bash
make eval   # re-ingests data/static/, writes data/eval/report.json
```

Embeddings, vector store, and LLM are all environment-config-driven
(ADR-001/002/003) — the [embedding model comparison](#embedding-model-comparison-multilingual-e5-base-vs-multilingual-e5-small)
below was produced with no code changes, only:

```bash
EMBEDDING_MODEL=intfloat/multilingual-e5-small \
  uv run python -m parking_bot.eval.harness --report data/eval/report_e5_small.json
```

## Results: `intfloat/multilingual-e5-base` (default, ADR-001)

Full report: [`data/eval/report.json`](../data/eval/report.json).

| K | Recall@K | Precision@K |
|---|---|---|
| 1 | 0.839 | 0.857 |
| 3 | 0.893 | 0.310 |
| 5 | 0.964 | 0.200 |
| 10 | 1.000 | 0.107 |

| Latency | Value |
|---|---|
| Mean | 59.5 ms |
| p50 | 58.9 ms |
| p95 | 67.0 ms |
| Min | 53.4 ms |
| Max | 74.5 ms |

## Embedding model comparison: `multilingual-e5-base` vs. `multilingual-e5-small`

Both are in the e5 family, so the `query:` / `passage:` prefix handling in
[`llm/embeddings.py`](../src/parking_bot/llm/embeddings.py) applies unchanged
to either — the swap is a one-line `.env` change, not a code change. Full
comparison report: [`data/eval/report_e5_small.json`](../data/eval/report_e5_small.json).

| K | Recall@K (base) | Recall@K (small) | Precision@K (base) | Precision@K (small) |
|---|---|---|---|---|
| 1 | 0.839 | 0.821 | 0.857 | 0.857 |
| 3 | 0.893 | 0.911 | 0.310 | 0.321 |
| 5 | 0.964 | 0.982 | 0.200 | 0.207 |
| 10 | 1.000 | 1.000 | 0.107 | 0.107 |

| | `multilingual-e5-base` | `multilingual-e5-small` |
|---|---|---|
| Dimensions | 768 | 384 |
| Mean latency | 59.5 ms | 24.2 ms |
| p95 latency | 67.0 ms | 26.6 ms |

**Reading this:** on this 28-question golden set the two models are within
noise of each other at K ≥ 3 (the small model is even marginally *higher* at
K=3/5, well within the margin a 28-question set can distinguish) while
`-small` is ~2.5× faster and stores half the vector dimensions. The one
place `-base` is clearly ahead is Recall@1 (0.839 vs. 0.821) — the case that
matters most since [`rag/chain.py`](../src/parking_bot/rag/chain.py) grounds
generation in whatever `retrieve()` returns, and `settings.top_k` defaults
to 4, close to where both curves have already converged (K=3 recall is
0.893/0.911).

This is the comparison ADR-001 argues for: because embeddings run locally
with no API key, comparing candidate models costs a few minutes of CPU time
and zero dollars, on real project data instead of a published benchmark.
That comparability is the reason ADR-001 chose a local model at all — which
specific local model wins is a secondary question this table exists to keep
revisiting cheaply as the golden set grows.

## Conclusions

- **Recall climbs fast, plateaus by K=5.** Every question's relevant
  document is retrievable by K=10, and 96% are already in by K=5 — the
  remaining gap is a small number of questions where the intended section
  isn't the nearest neighbor to the exact phrasing used in the golden set.
- **Precision@K falling as K grows is expected here, not a problem** — see
  Methodology. It says nothing about answer quality on its own; it's a
  byproduct of one-relevant-doc-per-question golden set entries and a fixed
  K denominator.
- **`top_k=4` (the current default) sits past the recall knee** (K=3 → 0.893,
  K=5 → 0.964), which is the right place to be for a grounded chain that
  needs the answer's section present in context far more than it needs a
  clean top-1 hit.
- **Model choice is close on this dataset; latency isn't.** If the golden
  set grows and keeps showing e5-small non-worse on recall, it's a
  reasonable default to switch to for the latency/footprint win — this
  report's methodology makes that an evidence-based call, not a guess.
- **Limitations:** 28 questions is a small sample (single-question swings
  move Recall@K by ~3.6 percentage points), there are no adversarial or
  out-of-scope questions in the set (see `rag/chain.py`'s grounding refusal
  behavior, which this harness doesn't score), and generation-quality
  (`expected_answer_contains`) isn't evaluated yet — only retrieval.
