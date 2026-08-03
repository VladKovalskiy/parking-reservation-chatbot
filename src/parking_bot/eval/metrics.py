"""Recall@K and Precision@K for retrieval evaluation against a golden set.

Both metrics take the *full* ranked list of retrieved doc_ids and a cutoff
`k`, rather than requiring the caller to pre-slice the list — that lets one
retrieval pass at `max(k for k in ks)` feed metrics at every smaller k.
"""

from collections.abc import Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of `relevant_doc_ids` that appear in the top-k retrieved results."""
    if not relevant_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    hits = sum(1 for doc_id in set(relevant_doc_ids) if doc_id in top_k)
    return hits / len(set(relevant_doc_ids))


def precision_at_k(
    retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int
) -> float:
    """Fraction of the top-k retrieved results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k
