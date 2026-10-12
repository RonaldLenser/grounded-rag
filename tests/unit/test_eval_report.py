"""Report math: errors count as failures, empty metrics become null, never NaN."""

import json
from pathlib import Path
from typing import Any

from evals.dataset import ExpectedSource, GoldenItem
from evals.records import ItemResult, SearchHit
from evals.report import load_run, summarize, write_report

ITEM = GoldenItem(
    id="q",
    category="answerable",
    question="q?",
    expected_sources=[ExpectedSource(file="a.pdf", pages=[1])],
    reference_answer="r",
    acceptable_statuses=["answered", "partial"],
)


def _answered() -> ItemResult:
    response: dict[str, Any] = {
        "status": "answered",
        "usage": {"llm_called": True, "model": "m", "input_tokens": 10, "output_tokens": 5},
        "verification": {"claims_supported": 1, "claims_rejected": 0, "downgraded_from": None},
        "citations": [],
        "latency_ms": 1000,
    }
    hit = SearchHit(filename="a.pdf", page_start=1, page_end=1, score=0.5)
    return ItemResult(item=ITEM, search_hits=[hit], response=response)


def test_errored_questions_count_as_failures() -> None:
    summary = summarize([_answered(), ItemResult(item=ITEM, error="502")])

    assert summary["counts"]["errors"] == 1
    assert summary["answers"]["answerable_answered"] == 0.5
    assert summary["answers"]["status_accuracy"] == 0.5


def test_report_round_trips_and_writes_strict_json(tmp_path: Path) -> None:
    meta = {
        "label": "t",
        "date": "2026-01-01 00:00 UTC",
        "target": "x",
        "collection": "c",
        "commit": "abc",
        "judge_model": None,
        "notes": "",
    }

    md_path, json_path = write_report([_answered()], out_dir=tmp_path, label="t", meta=meta)
    results, loaded_meta = load_run(json_path)

    text = json_path.read_text()
    assert "NaN" not in text  # no judge ran, so judge rates are null, not NaN
    assert json.loads(text)["summary"]["judge"]["claim_supported"] is None
    assert results[0].status == "answered" and loaded_meta == meta
    assert (
        "| Judge | Shown claims fully supported by their quotes | n/a of 0" in md_path.read_text()
    )
