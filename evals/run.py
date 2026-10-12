"""Run the golden set against a live Grounded instance and write a report.

Black-box on purpose: it talks to the public HTTP API, exactly like a user, so
it measures the deployed system (prompt, model, threshold, index), not a
re-implementation of it.

  uv run python -m evals.run                                  # local, no judge
  uv run python -m evals.run https://<service-url> --judge --label baseline
  uv run python -m evals.run --from-json docs/evals/<run>.json --judge   # judge a saved run

Per question: one /search (retrieval metrics + top score for gate calibration)
and one /query (the real answer). Requests are paced under the public rate
limits, and 429 responses are retried after Retry-After.
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx2
from anthropic import AsyncAnthropic

from evals.dataset import DEFAULT_GOLDEN_PATH, GoldenItem, load_golden
from evals.judge import Judge
from evals.records import ClaimJudgment, ItemResult, SearchHit
from evals.report import load_run, write_report
from grounded.providers.anthropic_llm import AnthropicLLM

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"
SEARCH_TOP_K = 6  # same as the app's RETRIEVAL_TOP_K, so retrieval metrics match /query
MAX_RETRIES = 5


class HttpClient(Protocol):
    """The subset of httpx2.Client we use (TestClient satisfies it too)."""

    def get(self, url: str) -> Any: ...
    def post(self, url: str, *, json: Any) -> Any: ...


class EvalAbortedError(Exception):
    pass


def find_collection(client: HttpClient, name: str) -> str:
    response = client.get("/v1/collections")
    response.raise_for_status()
    for collection in response.json():
        if collection["name"] == name:
            return str(collection["id"])
    raise EvalAbortedError(f"collection '{name}' not found: run scripts/seed_demo.py first")


def collect(
    client: HttpClient,
    items: list[GoldenItem],
    collection_id: str,
    *,
    query_interval: float = 6.5,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> list[ItemResult]:
    """Ask every golden question. Never raises for one bad question: errors are recorded."""
    results = []
    for number, item in enumerate(items, start=1):
        result = ItemResult(item=item)
        try:
            base = f"/v1/collections/{collection_id}"
            search = _post(
                client, f"{base}/search", {"query": item.question, "top_k": SEARCH_TOP_K}, sleep
            )
            result.search_hits = [SearchHit.model_validate(hit) for hit in search["results"]]
            result.response = _post(client, f"{base}/query", {"question": item.question}, sleep)
        except EvalAbortedError:
            raise
        except Exception as exc:  # recorded per question, so one failure can't sink a run
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            result.error = f"{type(exc).__name__}: {first_line}"[:300]
        results.append(result)
        log(f"  [{number:2d}/{len(items)}] {item.id:36s} {result.status or 'ERROR'}")
        if number < len(items) and query_interval > 0:
            sleep(query_interval)  # stay under the per-IP query rate limit
    return results


def _post(
    client: HttpClient, url: str, body: dict[str, Any], sleep: Callable[[float], None]
) -> Any:
    for _ in range(MAX_RETRIES):
        response = client.post(url, json=body)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        error = response.json().get("error", {})
        if error.get("code") == "daily_limit_reached":
            raise EvalAbortedError("the server's daily LLM budget is used up; try again tomorrow")
        sleep(float(response.headers.get("Retry-After", "10")) + 1)
    raise RuntimeError(f"still rate limited after {MAX_RETRIES} retries")


async def judge_all(
    results: list[ItemResult], judge: Judge, log: Callable[[str], None] = print
) -> None:
    """Fill in judge verdicts. Only shown claims and non-refused answers are judged.

    A judge failure is recorded on that question and the run continues: one bad
    judge call must not throw away every other verdict.
    """
    for result in results:
        response = result.response
        if response is None or response["status"] == "insufficient_context":
            continue
        try:
            await _judge_one(result, response, judge)
        except Exception as exc:  # recorded per question, like collection errors
            result.claim_judgments, result.answer_judgment = [], None
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            result.judge_error = f"{type(exc).__name__}: {first_line}"[:300]
            log(f"  judge FAILED {result.item.id}: {result.judge_error}")
            continue
        log(f"  judged {result.item.id}")


async def _judge_one(result: ItemResult, response: dict[str, Any], judge: Judge) -> None:
    quotes_by_source: dict[str, list[str]] = {}
    for citation in response["citations"]:
        quotes_by_source.setdefault(citation["source_id"], []).append(citation["quote"])
    for claim in response["claims"]:
        quotes = [q for sid in claim["source_ids"] for q in quotes_by_source.get(sid, [])]
        verdict = await judge.judge_claim(claim["text"], quotes)
        result.claim_judgments.append(
            ClaimJudgment(claim=claim["text"], verdict=verdict.verdict, reason=verdict.reason)
        )
    if result.item.reference_answer:
        result.answer_judgment = await judge.judge_answer(
            result.item.question, response["answer"], result.item.reference_answer
        )


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        result = subprocess.run(  # noqa: S603 (fixed arguments, no user input)
            [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _env(name: str) -> str | None:
    """Environment first, then .env (same convention as the smoke test)."""
    if value := os.environ.get(name):
        return value
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip()
    return None


def _build_judge(model: str) -> Judge:
    api_key = _env("ANTHROPIC_API_KEY")
    if not api_key:
        raise EvalAbortedError("--judge needs ANTHROPIC_API_KEY in the environment or .env")
    return Judge(AnthropicLLM(AsyncAnthropic(api_key=api_key, max_retries=3), model))


def _collect_live(args: argparse.Namespace) -> list[ItemResult]:
    items = load_golden(args.golden)[: args.limit]
    with httpx2.Client(base_url=args.base_url.rstrip("/"), timeout=120) as client:
        collection_id = find_collection(client, args.collection)
        print(f"Evaluating {len(items)} questions against {args.base_url}\n")
        return collect(client, items, collection_id, query_interval=args.query_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Grounded against the golden set.")
    parser.add_argument("base_url", nargs="?", default="http://localhost:8000")
    parser.add_argument("--collection", default="nist-ai-rmf")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--label", default="run", help="name used in the report file name")
    parser.add_argument("--judge", action="store_true", help="score claims and answers with an LLM")
    parser.add_argument("--judge-model", default=_env("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N questions (quick checks)")
    parser.add_argument(
        "--query-interval", type=float, default=6.5, help="seconds between questions"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("docs/evals"))
    parser.add_argument("--notes", default="", help="free text recorded in the report")
    parser.add_argument(
        "--from-json", type=Path, help="re-score (and optionally judge) a saved run, no API calls"
    )
    args = parser.parse_args(argv)

    started = datetime.now(UTC)
    try:
        judge = _build_judge(args.judge_model) if args.judge else None
        if args.from_json:
            results, meta = load_run(args.from_json)
        else:
            results = _collect_live(args)
            meta = {
                "label": args.label,
                "date": started.strftime("%Y-%m-%d %H:%M UTC"),
                "target": args.base_url,
                "collection": args.collection,
                "commit": _git_commit(),
                "judge_model": None,
                "notes": args.notes,
            }
        if judge is not None:
            # Save the collected answers first: if judging dies, nothing paid for is lost
            # (re-judge later with --from-json).
            write_report(results, out_dir=args.out_dir, label=meta["label"], meta=meta)
            print(f"\nSaved answers. Judging with {judge.model}")
            for result in results:  # re-judging replaces old verdicts
                result.claim_judgments, result.answer_judgment = [], None
                result.judge_error = None
            asyncio.run(judge_all(results, judge))
            meta["judge_model"] = judge.model
    except EvalAbortedError as exc:
        print(f"\nABORTED  {exc}", file=sys.stderr)
        return 1

    paths = write_report(results, out_dir=args.out_dir, label=meta["label"], meta=meta)
    print("\nWrote " + " and ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
