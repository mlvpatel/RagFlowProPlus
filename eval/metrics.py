"""Retrieval metrics for rag-agentic-2025 evaluation.

All functions operate on a ranked list of retrieved item ids (in rank order,
best first) and a set of relevant item ids. Pure and side effect free, so they
are unit tested without any database or model.
"""

from typing import Iterable, List, Set


def hit_at_k(ranked_ids: List, relevant: Set, k: int) -> float:
    """1.0 if any relevant id appears in the top k, else 0.0."""
    return 1.0 if any(i in relevant for i in ranked_ids[:k]) else 0.0


def precision_at_k(ranked_ids: List, relevant: Set, k: int) -> float:
    """Fraction of the top k retrieved items that are relevant."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for i in top_k if i in relevant) / len(top_k)


def recall_at_k(ranked_ids: List, relevant: Set, k: int) -> float:
    """Fraction of the relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    found = {i for i in ranked_ids[:k] if i in relevant}
    return len(found) / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(ranked_ids: List, relevant: Set) -> float:
    """1 divided by the rank of the first relevant item, else 0.0."""
    for index, item in enumerate(ranked_ids, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
