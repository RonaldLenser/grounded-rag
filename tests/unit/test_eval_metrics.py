"""Metric definitions are where evals silently lie, so each one is pinned by a test."""

import math
from typing import Any

import pytest

from evals import metrics as m
from evals.dataset import ExpectedSource, GoldenItem
from evals.records import ItemResult, SearchHit

SOURCE = ExpectedSource(file="rmf.pdf", pages=[12])


def _hit(
    filename: str = "rmf.pdf", start: int | None = 12, end: int | None = 12, score: float = 0.5
) -> SearchHit:
    return SearchHit(filename=filename, page_start=start, page_end=end, score=score)


def _item(category: str = "answerable", **extra: Any) -> GoldenItem:
    if category in ("answerable", "multi_source"):
        sources = extra.pop("sources", [SOURCE])
        return GoldenItem(
            id="q",
            category=category,  # type: ignore[arg-type]
            question="q?",
            expected_sources=sources,
            reference_answer="r",
            acceptable_statuses=["answered", "partial"],
        )
    return GoldenItem(
        id="q",
        category=category,  # type: ignore[arg-type]
        question="q?",
        acceptable_statuses=extra.pop("acceptable", ["insufficient_context"]),
    )


def _response(
    status: str, llm_called: bool = True, citations: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"status": status, "usage": {"llm_called": llm_called}, "citations": citations or []}


@pytest.mark.parametrize(
    ("hit", "expected"),
    [
        (_hit(start=12, end=12), True),
        (_hit(start=11, end=13), True),  # chunk spans the expected page
        (_hit(start=13, end=14), False),
        (_hit(filename="other.pdf"), False),
        (_hit(start=None, end=None), False),  # text files have no pages
        (_hit(start=12, end=None), True),
    ],
)
def test_page_matching(hit: SearchHit, expected: bool) -> None:
    assert m.matches(hit.filename, hit.page_start, hit.page_end, SOURCE) is expected


def test_rank_hit_at_k_and_mrr() -> None:
    hits = [_hit(start=1, end=1), _hit(start=12, end=12), _hit(start=12, end=12)]

    assert m.first_relevant_rank(hits, [SOURCE]) == 2
    assert m.first_relevant_rank(hits[:1], [SOURCE]) is None

    ranks: list[int | None] = [1, 2, None, 4]
    assert m.hit_at_k(ranks, 1) == 0.25
    assert m.hit_at_k(ranks, 3) == 0.5
    assert m.mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert math.isnan(m.mean_reciprocal_rank([]))


def test_multi_source_needs_every_document() -> None:
    other = ExpectedSource(file="genai.pdf", pages=[5])
    only_rmf = [_hit()]
    both = [_hit(), _hit(filename="genai.pdf", start=5, end=5)]

    assert not m.all_sources_retrieved(only_rmf, [SOURCE, other])
    assert m.all_sources_retrieved(both, [SOURCE, other])


def test_refusal_classification() -> None:
    gate = ItemResult(
        item=_item("out_of_domain"), response=_response("insufficient_context", llm_called=False)
    )
    model = ItemResult(item=_item("unanswerable"), response=_response("insufficient_context"))
    wrong = ItemResult(item=_item("unanswerable"), response=_response("answered"))
    failed = ItemResult(item=_item(), error="boom")

    assert m.gate_refused(gate) and m.status_ok(gate)
    assert m.refused(model) and not m.gate_refused(model) and m.status_ok(model)
    assert not m.status_ok(wrong)
    assert not m.status_ok(failed), "an errored question never counts as correct"


def test_citation_page_hit() -> None:
    cite = {"filename": "rmf.pdf", "page_start": 12, "page_end": 12}
    off = {"filename": "rmf.pdf", "page_start": 30, "page_end": 30}

    assert m.citation_page_hit(
        ItemResult(item=_item(), response=_response("answered", citations=[off, cite]))
    )
    assert not m.citation_page_hit(
        ItemResult(item=_item(), response=_response("answered", citations=[off]))
    )


def _scored(category: str, score: float) -> ItemResult:
    return ItemResult(
        item=_item(category), search_hits=[_hit(score=score)], response=_response("answered")
    )


def test_threshold_sweep_counts_what_the_gate_would_block() -> None:
    results = [
        _scored("answerable", 0.40),
        _scored("answerable", 0.30),
        _scored("out_of_domain", 0.10),
    ]

    rows = {row.threshold: row for row in m.threshold_sweep(results)}

    assert (rows[0.10].answerable_blocked, rows[0.10].refusable_blocked) == (0, 0)
    assert (rows[0.15].answerable_blocked, rows[0.15].refusable_blocked) == (0, 1)
    assert (rows[0.35].answerable_blocked, rows[0.35].refusable_blocked) == (1, 1)


@pytest.mark.parametrize(
    ("lowest_answerable", "expected"),
    [(0.31, 0.15), (0.41, 0.20), (0.12, 0.05), (0.03, 0.0)],
)
def test_recommended_threshold_sits_in_the_gap(lowest_answerable: float, expected: float) -> None:
    results = [
        _scored("answerable", lowest_answerable),
        _scored("answerable", 0.6),
        _scored("out_of_domain", 0.01),
    ]

    recommended = m.recommend_threshold(results)

    assert recommended == expected
    assert recommended is not None and recommended < lowest_answerable


def test_threshold_falls_back_below_answerable_when_groups_overlap() -> None:
    results = [_scored("answerable", 0.31), _scored("out_of_domain", 0.50)]

    assert m.recommend_threshold(results) == 0.25


def test_threshold_from_the_baseline_run_scores() -> None:
    """M7 baseline: highest off-topic 0.14, lowest answerable 0.31 -> middle 0.225 -> 0.20."""
    results = [_scored("answerable", 0.31), _scored("out_of_domain", 0.14)]

    assert m.recommend_threshold(results) == 0.20


def test_percentile_is_nearest_rank() -> None:
    values = [100.0, 200.0, 300.0, 400.0]

    assert m.percentile(values, 50) == 200.0
    assert m.percentile(values, 95) == 400.0
    assert math.isnan(m.percentile([], 50))
