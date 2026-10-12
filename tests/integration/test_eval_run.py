"""The whole harness, end to end: real app + database over HTTP, fake model and judge."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from evals.dataset import ExpectedSource, GoldenItem
from evals.judge import Judge
from evals.report import write_report
from evals.run import EvalAbortedError, _post, collect, find_collection, judge_all
from tests.unit.test_eval_judge import ScriptedLLM

PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"

ITEMS = [
    GoldenItem(
        id="measure-tools",
        category="answerable",
        question="What does the MEASURE function use to analyze AI risk?",
        expected_sources=[ExpectedSource(file="sample.pdf", pages=[2])],
        reference_answer="Quantitative tools.",
        acceptable_statuses=["answered", "partial"],
    ),
    GoldenItem(
        id="cookies",
        category="out_of_domain",
        question="best chocolate chip cookie recipe",
        acceptable_statuses=["insufficient_context"],
    ),
]


def _seed(api: TestClient) -> str:
    collection_id = str(api.post("/v1/collections", json={"name": "eval-demo"}).json()["id"])
    upload = api.post(
        f"/v1/collections/{collection_id}/documents",
        files={"file": ("sample.pdf", PDF.read_bytes(), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    return collection_id


async def test_full_run_writes_a_report(api: TestClient, tmp_path: Path) -> None:
    collection_id = _seed(api)
    assert find_collection(api, "eval-demo") == collection_id

    results = collect(api, ITEMS, collection_id, query_interval=0, log=lambda _: None)
    judge = Judge(
        ScriptedLLM(
            {
                "claim_support": {"verdict": "supported", "reason": "Quote states it."},
                "answer_correctness": {"verdict": "correct", "reason": "Matches."},
            }
        )
    )
    await judge_all(results, judge, log=lambda _: None)
    md_path, json_path = write_report(
        results,
        out_dir=tmp_path,
        label="Test Run!",
        meta={
            "label": "Test Run!",
            "date": "2026-10-12 12:00 UTC",
            "target": "testclient",
            "collection": "eval-demo",
            "commit": "abc1234",
            "judge_model": judge.model,
            "notes": "",
        },
    )

    answered, refused = results
    assert answered.status == "answered" and answered.search_hits
    assert answered.claim_judgments and answered.answer_judgment
    assert refused.status == "insufficient_context"
    assert refused.answer_judgment is None, "refusals are not judged"

    assert md_path.name == "2026-10-12-test-run.md"
    report = md_path.read_text()
    assert (
        "| Retrieval | Hit@1 / Hit@3 / Hit@6 (expected page retrieved) | 100% / 100% / 100% |"
        in report
    )
    assert "| Answers | Out-of-domain refused by the gate (no LLM call) | 100% |" in report
    raw = json.loads(json_path.read_text())  # strict JSON: no NaN anywhere
    assert raw["summary"]["counts"]["total"] == 2
    assert raw["summary"]["judge"]["claim_supported"] == 1.0


def test_one_failing_question_does_not_sink_the_run(api: TestClient) -> None:
    results = collect(
        api, ITEMS[:1], "00000000-0000-0000-0000-000000000000", query_interval=0, log=lambda _: None
    )

    [result] = results
    assert result.error is not None and "404" in result.error
    assert result.response is None


def test_missing_collection_aborts(api: TestClient) -> None:
    with pytest.raises(EvalAbortedError, match="seed_demo"):
        find_collection(api, "does-not-exist")


class _Response:
    def __init__(
        self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> None:
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        assert self.status_code < 400


class _ScriptedClient:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses

    def get(self, url: str) -> _Response:
        raise NotImplementedError

    def post(self, url: str, *, json: Any) -> _Response:
        return self.responses.pop(0)


def test_rate_limits_are_waited_out_using_retry_after() -> None:
    slept: list[float] = []
    client = _ScriptedClient(
        [
            _Response(429, {"error": {"code": "rate_limited"}}, {"Retry-After": "7"}),
            _Response(200, {"ok": True}),
        ]
    )

    assert _post(client, "/x", {}, slept.append) == {"ok": True}
    assert slept == [8.0]


def test_daily_budget_exhaustion_aborts_instead_of_retrying() -> None:
    client = _ScriptedClient([_Response(429, {"error": {"code": "daily_limit_reached"}})])

    with pytest.raises(EvalAbortedError, match="daily"):
        _post(client, "/x", {}, lambda _: None)


class _FlakyJudgeLLM(ScriptedLLM):
    """Fails on the answer-correctness call, like a judge reply cut off at max_tokens."""

    async def complete_structured(self, **kwargs: Any) -> Any:
        if kwargs["output"].name == "answer_correctness":
            raise RuntimeError("judge reply cut off")
        return await super().complete_structured(**kwargs)


async def test_a_judge_failure_is_recorded_and_the_run_continues(
    api: TestClient, tmp_path: Path
) -> None:
    collection_id = _seed(api)
    results = collect(api, ITEMS, collection_id, query_interval=0, log=lambda _: None)
    llm = _FlakyJudgeLLM({"claim_support": {"verdict": "supported", "reason": "ok"}})

    await judge_all(results, Judge(llm), log=lambda _: None)

    answered = results[0]
    assert answered.judge_error == "RuntimeError: judge reply cut off"
    assert answered.claim_judgments == [], "no half-judged question is counted"
    assert answered.answer_judgment is None
