from parking_bot.eval.metrics import precision_at_k, recall_at_k


def test_recall_at_k_counts_relevant_docs_found_within_the_cutoff() -> None:
    retrieved = ["a", "b", "c"]
    relevant = ["b", "x"]

    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_at_k_ignores_hits_beyond_the_cutoff() -> None:
    retrieved = ["x", "a", "b"]
    relevant = ["b"]

    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_with_no_relevant_docs_is_zero() -> None:
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_precision_at_k_counts_relevant_hits_over_the_cutoff() -> None:
    retrieved = ["a", "b", "c"]
    relevant = ["b"]

    assert precision_at_k(retrieved, relevant, k=3) == 1 / 3


def test_precision_at_k_uses_k_as_the_denominator_not_len_retrieved() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["a", "b", "c", "d"]

    assert precision_at_k(retrieved, relevant, k=2) == 1.0
    assert precision_at_k(retrieved, relevant, k=4) == 1.0


def test_precision_at_k_with_zero_k_is_zero() -> None:
    assert precision_at_k(["a"], ["a"], k=0) == 0.0
