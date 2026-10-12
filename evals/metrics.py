"""Scoring: pure functions from ItemResults to numbers. No I/O, fully unit-tested.

Definitions (also printed in every report):
  Hit@k        an expected page appears in the top k retrieved chunks
  MRR          mean of 1/rank of the first relevant chunk (0 if none in top k)
  Status OK    the final status is one the golden item accepts
  False refusal  an answerable question got insufficient_context
  Gate refusal   refused without calling the LLM (cost $0)
  Citation page hit  at least one shown citation points at an expected page
  Claim support  share of SHOWN claims the judge rates fully supported by their quotes
"""

import math
from dataclasses import dataclass
from statistics import median

from evals.dataset import ExpectedSource
from evals.records import ItemResult, SearchHit

THRESHOLD_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


# ----- Retrieval -----
def matches(
    filename: str, page_start: int | None, page_end: int | None, source: ExpectedSource
) -> bool:
    if filename != source.file or page_start is None:
        return False
    end = page_end if page_end is not None else page_start
    return any(page_start <= page <= end for page in source.pages)


def first_relevant_rank(hits: list[SearchHit], sources: list[ExpectedSource]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(matches(hit.filename, hit.page_start, hit.page_end, s) for s in sources):
            return rank
    return None


def all_sources_retrieved(hits: list[SearchHit], sources: list[ExpectedSource]) -> bool:
    """Multi-source questions: every expected document contributed a relevant chunk."""
    return all(any(matches(h.filename, h.page_start, h.page_end, s) for h in hits) for s in sources)


def hit_at_k(ranks: list[int | None], k: int) -> float:
    return _rate([rank is not None and rank <= k for rank in ranks])


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    return sum(1 / rank for rank in ranks if rank) / len(ranks) if ranks else math.nan


# ----- Answers -----
def status_ok(result: ItemResult) -> bool:
    return result.status in result.item.acceptable_statuses


def refused(result: ItemResult) -> bool:
    return result.status == "insufficient_context"


def gate_refused(result: ItemResult) -> bool:
    """Refused before any LLM call: the relevance gate, not the model, said no."""
    return (
        refused(result)
        and result.response is not None
        and not result.response["usage"]["llm_called"]
    )


def citation_page_hit(result: ItemResult) -> bool:
    if result.response is None:
        return False
    return any(
        matches(c["filename"], c["page_start"], c["page_end"], source)
        for c in result.response["citations"]
        for source in result.item.expected_sources
    )


# ----- Relevance gate calibration -----
@dataclass(frozen=True)
class GateRow:
    threshold: float
    answerable_blocked: int  # false refusals the gate would cause
    answerable_total: int
    refusable_blocked: int  # unanswerable/out-of-domain refused for free
    refusable_total: int


def threshold_sweep(
    results: list[ItemResult], thresholds: tuple[float, ...] = THRESHOLD_SWEEP
) -> list[GateRow]:
    """Simulate the gate at each threshold using each question's top-1 retrieval score."""
    scored = [r for r in results if r.top_score is not None]
    answer = [r.top_score for r in scored if r.item.should_answer]
    refuse = [r.top_score for r in scored if not r.item.should_answer]
    return [
        GateRow(
            threshold=t,
            answerable_blocked=sum(s is not None and s < t for s in answer),
            answerable_total=len(answer),
            refusable_blocked=sum(s is not None and s < t for s in refuse),
            refusable_total=len(refuse),
        )
        for t in thresholds
    ]


def recommend_threshold(results: list[ItemResult], fallback_margin: float = 0.05) -> float | None:
    """Max-margin threshold: the middle of the gap between the highest out-of-domain
    score and the lowest answerable score, rounded down to a multiple of 0.05.

    Sitting in the middle leaves equal room for unseen questions on both sides.
    Rounding down favors answering: a false refusal (a real answer withheld) is the
    costlier error, because refusable questions that slip past the gate still meet
    the model's own refusal and the claim verifier. If the two groups overlap
    there is no clean gap, so fall back to just below the lowest answerable score.
    """
    answerable = [r.top_score for r in results if r.item.should_answer and r.top_score is not None]
    off_topic = [
        r.top_score
        for r in results
        if r.item.category == "out_of_domain" and r.top_score is not None
    ]
    if not answerable:
        return None
    lowest = min(answerable)
    highest_off_topic = max(off_topic, default=0.0)
    target = (
        (lowest + highest_off_topic) / 2 if highest_off_topic < lowest else lowest - fallback_margin
    )
    # round() first so float noise (0.35 - 0.05 = 0.29999...) does not drop a whole step
    return max(math.floor(round(target * 20, 6)) / 20, 0.0)


# ----- Shared helpers -----
def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation): honest for small samples."""
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(math.ceil(pct / 100 * len(ordered)) - 1, 0)
    return ordered[index]


def score_summary(values: list[float]) -> tuple[float, float, float]:
    """(min, median, max), or NaNs when empty."""
    return (min(values), median(values), max(values)) if values else (math.nan,) * 3


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else math.nan
