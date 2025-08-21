from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from askdocs.rag.store import Hit


def mmr(query: Sequence[float], hits: Sequence[Hit], k: int, lambda_: float = 0.7) -> list[Hit]:
    """Maximal marginal relevance: trade relevance for diversity among the fetched hits.

    Vectors are expected to be L2-normalized, so dot product is cosine similarity.
    """
    if len(hits) <= 1 or k <= 0:
        return list(hits[:k])

    q = np.asarray(query, dtype=np.float32)
    docs = np.asarray([h.vector for h in hits], dtype=np.float32)
    relevance = docs @ q
    similarity = docs @ docs.T

    selected = [int(np.argmax(relevance))]
    candidates = set(range(len(hits))) - set(selected)
    while candidates and len(selected) < k:
        best, best_score = -1, -np.inf
        for i in candidates:
            redundancy = similarity[i, selected].max()
            score = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        candidates.remove(best)
    return [hits[i] for i in selected]


def fit_to_budget(hits: Sequence[Hit], max_chars: int) -> list[Hit]:
    """Keep as many hits as fit into max_chars, preserving order.

    The first hit is always kept, trimmed if it alone is over the budget.
    """
    result: list[Hit] = []
    used = 0
    for hit in hits:
        if used + len(hit.text) > max_chars:
            if not result:
                trimmed = hit.text[:max_chars].rsplit(" ", 1)[0]
                result.append(replace(hit, text=trimmed))
                used = len(trimmed)
            continue
        result.append(hit)
        used += len(hit.text)
    return result
