"""Reciprocal Rank Fusion: the merge step that decides hybrid result order."""

import uuid

import pytest

from grounded.retrieval.fusion import reciprocal_rank_fusion
from grounded.retrieval.retriever import RetrievedChunk, best_similarity

A, B, C, D = (uuid.UUID(int=i) for i in range(1, 5))


def test_items_found_by_both_lists_rise_to_the_top() -> None:
    fused = reciprocal_rank_fusion([[A, B, C], [C, D]], k=60)

    assert [f.item_id for f in fused] == [C, A, B, D]
    assert fused[0].ranks == (3, 1)
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)


def test_scores_use_ranks_not_raw_values() -> None:
    """Only order matters: the same order gives the same fusion, whatever the scores were."""
    fused = reciprocal_rank_fusion([[A, B], []], k=60)

    assert [(f.item_id, f.ranks) for f in fused] == [(A, (1, None)), (B, (2, None))]
    assert fused[0].score == pytest.approx(1 / 61)


def test_ties_break_by_best_single_rank_then_input_order() -> None:
    # A is 1st in list one, D is 1st in list two: equal scores, equal best rank.
    fused = reciprocal_rank_fusion([[A], [D]], k=60)

    assert [f.item_id for f in fused] == [A, D]


def test_duplicates_within_a_list_keep_their_best_rank() -> None:
    fused = reciprocal_rank_fusion([[A, B, A]], k=60)

    assert {f.item_id: f.ranks for f in fused} == {A: (1,), B: (2,)}


def test_small_k_lets_one_first_place_win_large_k_rewards_agreement() -> None:
    """Why k matters. A is 1st in one list only; B is 4th in both lists."""
    x, y, z, w = (uuid.UUID(int=i) for i in range(10, 14))
    lists = [[A, x, y, B], [D, z, w, B]]

    assert reciprocal_rank_fusion(lists, k=1)[0].item_id == A  # 1/2 beats 2 * 1/5
    assert reciprocal_rank_fusion(lists, k=60)[0].item_id == B  # 2 * 1/64 beats 1/61


def test_empty_input_and_invalid_k() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[A]], k=0)


def _chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="f",
        chunk_index=0,
        content="c",
        page_start=None,
        page_end=None,
        heading=None,
        score=score,
    )


def test_gate_input_is_the_best_similarity_not_the_first_result() -> None:
    """Hybrid orders by fused rank, so the first result may not have the best cosine."""
    assert best_similarity([_chunk(0.12), _chunk(0.55), _chunk(0.30)]) == 0.55
    assert best_similarity([]) is None
