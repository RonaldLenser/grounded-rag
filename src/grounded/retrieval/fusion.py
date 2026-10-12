"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

Each list votes for its items with 1 / (k + rank). Items ranked well by more
than one list rise to the top. Only RANKS are used, never raw scores, which is
the point: cosine similarity (0..1) and keyword weights (0..~20) live on
different scales and can't be added directly.

k (60 by convention, from the original RRF paper) damps the difference between
rank 1 and rank 2 so that one list can't dominate on a single strong hit.
"""

import uuid
from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedItem:
    item_id: uuid.UUID
    score: float
    ranks: tuple[int | None, ...]  # 1-based rank in each input list, None if absent


def reciprocal_rank_fusion(
    rankings: list[list[uuid.UUID]], k: int = DEFAULT_RRF_K
) -> list[FusedItem]:
    """Fuse ranked lists (best first). Ties break by best single rank, then input order."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranks: dict[uuid.UUID, list[int | None]] = {}
    for list_index, ranking in enumerate(rankings):
        for rank, item_id in enumerate(ranking, start=1):
            slot = ranks.setdefault(item_id, [None] * len(rankings))
            if slot[list_index] is None:  # duplicates within one list: keep the best rank
                slot[list_index] = rank

    order = {item_id: position for position, item_id in enumerate(ranks)}
    fused = [
        FusedItem(
            item_id=item_id,
            score=sum(1 / (k + r) for r in item_ranks if r is not None),
            ranks=tuple(item_ranks),
        )
        for item_id, item_ranks in ranks.items()
    ]
    fused.sort(
        key=lambda f: (
            -f.score,
            min(r for r in f.ranks if r is not None),
            order[f.item_id],
        )
    )
    return fused
