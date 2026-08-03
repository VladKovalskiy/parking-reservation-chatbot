"""Evaluation harness: run data/eval/golden_set.jsonl through retrieval and
compute Recall@K / Precision@K for one or more values of K.

Run:  make eval
      uv run python -m parking_bot.eval.harness
      uv run python -m parking_bot.eval.harness --k 1 3 5 10 --report data/eval/report.json

Re-ingests data/static/ before evaluating by default (pass --skip-ingest to
reuse whatever is already indexed). Uses whichever embedding provider and
Milvus target are configured in .env (ADR-001/002/003), so re-running this
against a different embedding model is a config change, not a code change.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from parking_bot.config import Settings, get_settings
from parking_bot.eval.metrics import precision_at_k, recall_at_k
from parking_bot.retrieval.retriever import retrieve
from parking_bot.retrieval.store import build_vector_store

DEFAULT_GOLDEN_SET = Path("data/eval/golden_set.jsonl")
DEFAULT_REPORT_PATH = Path("data/eval/report.json")
DEFAULT_STATIC_DIR = Path("data/static")
DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class GoldenExample:
    question: str
    relevant_doc_ids: list[str]


def load_golden_set(path: Path) -> list[GoldenExample]:
    """Parse golden_set.jsonl into GoldenExamples (ignores blank lines)."""
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append(
            GoldenExample(question=row["question"], relevant_doc_ids=row["relevant_doc_ids"])
        )
    return examples


def run_eval(
    golden_set: list[GoldenExample],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    store=None,
    settings: Settings | None = None,
) -> dict:
    """Retrieve for every question once (at max(ks)) and score at each k in ks."""
    settings = settings or get_settings()
    store = store or build_vector_store(settings)
    max_k = max(ks)

    recalls: dict[int, list[float]] = {k: [] for k in ks}
    precisions: dict[int, list[float]] = {k: [] for k in ks}

    for example in golden_set:
        chunks = retrieve(example.question, store=store, k=max_k, settings=settings)
        retrieved_doc_ids = [chunk.metadata["doc_id"] for chunk in chunks]
        for k in ks:
            recalls[k].append(recall_at_k(retrieved_doc_ids, example.relevant_doc_ids, k))
            precisions[k].append(precision_at_k(retrieved_doc_ids, example.relevant_doc_ids, k))

    return {
        "num_questions": len(golden_set),
        "k_values": list(ks),
        "metrics": {
            str(k): {
                "recall_at_k": mean(recalls[k]) if golden_set else 0.0,
                "precision_at_k": mean(precisions[k]) if golden_set else 0.0,
            }
            for k in ks
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="reuse whatever is already indexed instead of re-ingesting data/static/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    if not args.skip_ingest:
        from parking_bot.ingestion.pipeline import run_ingestion

        chunk_count = run_ingestion(settings, static_dir=args.static_dir)
        print(f"[ingest] {chunk_count} chunks indexed from {args.static_dir}")

    golden_set = load_golden_set(args.golden_set)
    print(f"[eval] running {len(golden_set)} questions through retrieval for k={args.k}")

    report = run_eval(golden_set, ks=args.k, settings=settings)
    report["embedding_provider"] = settings.embedding_provider
    report["embedding_model"] = settings.embedding_model

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote report to {args.report}")

    for k in args.k:
        m = report["metrics"][str(k)]
        print(f"  k={k:<3} recall@k={m['recall_at_k']:.3f}  precision@k={m['precision_at_k']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
