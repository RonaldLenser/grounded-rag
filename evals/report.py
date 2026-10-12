"""Turn scored results into a committed Markdown report (+ raw JSON for auditing)."""

import json
import math
import re
from pathlib import Path
from typing import Any

from evals import metrics as m
from evals.records import ItemResult


def summarize(results: list[ItemResult]) -> dict[str, Any]:
    # Errored questions stay IN every denominator and count as failures: dropping
    # them would make a flaky system look better than a reliable one.
    ok = [r for r in results if r.response is not None]
    answerable = [r for r in results if r.item.should_answer]
    unanswerable = [r for r in results if r.item.category == "unanswerable"]
    out_of_domain = [r for r in results if r.item.category == "out_of_domain"]
    multi = [r for r in results if r.item.category == "multi_source"]
    llm_calls = [r for r in ok if r.response and r.response["usage"]["llm_called"]]

    ranks = [m.first_relevant_rank(r.search_hits, r.item.expected_sources) for r in answerable]
    answered = [r for r in answerable if r.response is not None and not m.refused(r)]
    verification = [r.response["verification"] for r in ok if r.response]
    claims_shown = sum(v["claims_supported"] for v in verification)
    claims_dropped = sum(v["claims_rejected"] for v in verification)
    judged_claims = [j for r in ok for j in r.claim_judgments]
    judged_answers = [r.answer_judgment for r in answerable if r.answer_judgment]

    return {
        "counts": {
            "total": len(results),
            "errors": len(results) - len(ok),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "out_of_domain": len(out_of_domain),
        },
        "retrieval": {
            "hit_at_1": m.hit_at_k(ranks, 1),
            "hit_at_3": m.hit_at_k(ranks, 3),
            "hit_at_6": m.hit_at_k(ranks, 6),
            "mrr": m.mean_reciprocal_rank(ranks),
            "multi_source_full_coverage": _rate(
                [m.all_sources_retrieved(r.search_hits, r.item.expected_sources) for r in multi]
            ),
        },
        "answers": {
            "status_accuracy": _rate([m.status_ok(r) for r in results]),
            "answerable_answered": _rate(
                [r.response is not None and not m.refused(r) for r in answerable]
            ),
            "false_refusals": sum(m.refused(r) for r in answerable),
            "false_refusals_retrieval_miss": sum(
                m.refused(r) and rank is None for r, rank in zip(answerable, ranks, strict=True)
            ),
            "answerable_fully_answered": _rate([r.status == "answered" for r in answerable]),
            "unanswerable_refused_ok": _rate([m.status_ok(r) for r in unanswerable]),
            "unanswerable_strict_refusal": _rate([m.refused(r) for r in unanswerable]),
            "out_of_domain_gate_refused": _rate([m.gate_refused(r) for r in out_of_domain]),
            "citation_page_hit": _rate([m.citation_page_hit(r) for r in answered]),
        },
        "verifier": {
            "claims_proposed": claims_shown + claims_dropped,
            "claims_shown": claims_shown,
            "claims_dropped": claims_dropped,
            "drop_rate": claims_dropped / (claims_shown + claims_dropped)
            if claims_shown + claims_dropped
            else math.nan,
            "downgraded": sum(v["downgraded_from"] is not None for v in verification),
        },
        "judge": {
            "claims_judged": len(judged_claims),
            "judge_errors": sum(r.judge_error is not None for r in results),
            "claim_supported": _rate([j.verdict == "supported" for j in judged_claims]),
            "claim_not_supported": sum(j.verdict == "not_supported" for j in judged_claims),
            "answers_judged": len(judged_answers),
            "answer_correct": _rate([a.verdict == "correct" for a in judged_answers]),
            "answer_correct_or_partial": _rate([a.verdict != "incorrect" for a in judged_answers]),
        },
        "cost_latency": {
            "llm_calls": len(llm_calls),
            "latency_p50_ms": m.percentile(
                [r.response["latency_ms"] for r in llm_calls if r.response], 50
            ),
            "latency_p95_ms": m.percentile(
                [r.response["latency_ms"] for r in llm_calls if r.response], 95
            ),
            "avg_input_tokens": _mean(
                [r.response["usage"]["input_tokens"] for r in llm_calls if r.response]
            ),
            "avg_output_tokens": _mean(
                [r.response["usage"]["output_tokens"] for r in llm_calls if r.response]
            ),
        },
        "gate": {
            "answerable_top_scores": m.score_summary(
                [r.top_score for r in answerable if r.top_score is not None]
            ),
            "unanswerable_top_scores": m.score_summary(
                [r.top_score for r in unanswerable if r.top_score is not None]
            ),
            "out_of_domain_top_scores": m.score_summary(
                [r.top_score for r in out_of_domain if r.top_score is not None]
            ),
            "sweep": [row.__dict__ for row in m.threshold_sweep(ok)],
            "recommended_threshold": m.recommend_threshold(ok),
        },
        "models": sorted({str(r.response["usage"]["model"]) for r in llm_calls if r.response}),
        "strategies": sorted({r.strategy for r in results if r.strategy}),
    }


def load_run(path: Path) -> tuple[list[ItemResult], dict[str, Any]]:
    """Read a saved run back, to re-score or re-judge it without calling any API."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ItemResult.model_validate(r) for r in raw["results"]], raw["meta"]


def write_report(
    results: list[ItemResult], *, out_dir: Path, label: str, meta: dict[str, Any]
) -> tuple[Path, Path]:
    summary = summarize(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['date'][:10]}-{re.sub(r'[^a-z0-9-]+', '-', label.lower()).strip('-')}"
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"
    md_path.write_text(render_markdown(summary, results, meta), encoding="utf-8")
    raw = {
        "meta": meta,
        "summary": summary,
        "results": [r.model_dump(mode="json") for r in results],
    }
    json_path.write_text(json.dumps(_finite(raw), indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def render_markdown(
    summary: dict[str, Any], results: list[ItemResult], meta: dict[str, Any]
) -> str:
    c, gate = summary["counts"], summary["gate"]
    n_ans = c["answerable"]
    lines = [
        f"# Grounded evaluation: {meta.get('label', '')}".rstrip(": "),
        "",
        f"- **Date:** {meta['date']}",
        f"- **Target:** {meta['target']} "
        f"(collection `{meta['collection']}`, commit `{meta['commit']}`)",
        f"- **Retrieval:** {', '.join(summary.get('strategies') or ['vector (pre-M8 server)'])}",
        f"- **Generation model:** {', '.join(summary['models']) or 'n/a'}",
        f"- **Judge model:** {meta['judge_model'] or 'not run (pass --judge)'}",
        f"- **Golden set:** {c['total']} questions ({n_ans} answerable incl. multi-source, "
        f"{c['unanswerable']} unanswerable, {c['out_of_domain']} out-of-domain); "
        f"{c['errors']} error{'' if c['errors'] == 1 else 's'}",
    ]
    if meta.get("notes"):
        lines.append(f"- **Notes:** {meta['notes']}")
    lines += [
        "",
        "## Summary",
        "",
        "| Area | Metric | Result |",
        "|---|---|---|",
        *(f"| {area} | {metric} | {value} |" for area, metric, value in _summary_rows(summary)),
        "",
        "## Relevance gate calibration",
        "",
        "Top-1 retrieval similarity per question (min / median / max):",
        "",
        "| Category | Min | Median | Max |",
        "|---|---|---|---|",
    ]
    for name, key in (
        ("Answerable + multi-source", "answerable_top_scores"),
        ("Unanswerable (on-topic)", "unanswerable_top_scores"),
        ("Out-of-domain", "out_of_domain_top_scores"),
    ):
        lo, mid, hi = gate[key]
        lines.append(f"| {name} | {_num(lo)} | {_num(mid)} | {_num(hi)} |")
    lines += [
        "",
        "If the gate threshold were T, it would refuse (without calling the LLM):",
        "",
        "| T | Answerable wrongly refused | Unanswerable/out-of-domain refused for free |",
        "|---|---|---|",
    ]
    for row in gate["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['answerable_blocked']}/{row['answerable_total']} | "
            f"{row['refusable_blocked']}/{row['refusable_total']} |"
        )
    rec = gate["recommended_threshold"]
    lines += [
        "",
        f"Recommended `MIN_RELEVANCE_SCORE`: **{_num(rec)}** (middle of the gap between the "
        "highest out-of-domain and lowest answerable score, rounded down to 0.05). On-topic "
        "unanswerable questions score like answerable ones, so no threshold can catch them; "
        "the model's refusal and the claim verifier do.",
        "",
        "## Per-question results",
        "",
        "| Question | Category | Status | OK | Top score | First relevant rank "
        "| Claims shown/dropped | Judge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rank = (
            m.first_relevant_rank(r.search_hits, r.item.expected_sources)
            if r.item.should_answer
            else None
        )
        verification = r.response["verification"] if r.response else None
        claims = (
            f"{verification['claims_supported']}/{verification['claims_rejected']}"
            if verification
            else "-"
        )
        judge = r.answer_judgment.verdict if r.answer_judgment else "-"
        lines.append(
            f"| `{r.item.id}` | {r.item.category} | {r.status or 'ERROR'} | "
            f"{'yes' if m.status_ok(r) else '**no**'} | {_num(r.top_score)} | "
            f"{rank if rank else ('miss' if r.item.should_answer else '-')} | {claims} | {judge} |"
        )
    problems = _problems(results)
    if problems:
        lines += ["", "## Failures and judge notes", "", *problems]
    lines += [
        "",
        "## Method",
        "",
        "- Golden set: `evals/golden.jsonl`, hand-written from the two NIST PDFs. Every expected "
        "page was checked to contain the answer; every unanswerable question was checked by text "
        "search to be absent from both documents.",
        "- Each question is sent once to `/search` (top 6, for retrieval and gate metrics) and "
        "once to `/query` (the real answer path), over HTTP against the target.",
        "- Judge verdicts come from an LLM and can be wrong. Each one includes a reason below or "
        "in the JSON file; spot-check before quoting numbers.",
        "- Small sample: one question is "
        f"{_pct(1 / n_ans) if n_ans else 'n/a'} of the answerable set. Treat differences of one "
        "or two questions as noise.",
    ]
    return "\n".join(lines) + "\n"


def _summary_rows(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    c, ret, ans, ver, jd, cl = (
        summary[k] for k in ("counts", "retrieval", "answers", "verifier", "judge", "cost_latency")
    )
    hits = " / ".join(_pct(ret[f"hit_at_{k}"]) for k in (1, 3, 6))
    return [
        ("Retrieval", "Hit@1 / Hit@3 / Hit@6 (expected page retrieved)", hits),
        ("Retrieval", "MRR", _num(ret["mrr"])),
        (
            "Retrieval",
            "Multi-source: both documents retrieved",
            _pct(ret["multi_source_full_coverage"]),
        ),
        ("Answers", "Correct status (all questions)", _pct(ans["status_accuracy"])),
        (
            "Answers",
            "Answerable questions answered (not refused)",
            f"{_pct(ans['answerable_answered'])} ({ans['false_refusals']} false refusals, "
            f"{ans['false_refusals_retrieval_miss']} caused by a retrieval miss)",
        ),
        (
            "Answers",
            "Answerable questions with status `answered`",
            _pct(ans["answerable_fully_answered"]),
        ),
        (
            "Answers",
            "Unanswerable: no unsupported answer (refused or partial)",
            f"{_pct(ans['unanswerable_refused_ok'])} "
            f"(strict refusal {_pct(ans['unanswerable_strict_refusal'])})",
        ),
        (
            "Answers",
            "Out-of-domain refused by the gate (no LLM call)",
            _pct(ans["out_of_domain_gate_refused"]),
        ),
        ("Citations", "Answered questions citing an expected page", _pct(ans["citation_page_hit"])),
        (
            "Verifier",
            "Claims proposed / shown / dropped",
            f"{ver['claims_proposed']} / {ver['claims_shown']} / {ver['claims_dropped']} "
            f"({_pct(ver['drop_rate'])} dropped)",
        ),
        (
            "Judge",
            "Shown claims fully supported by their quotes",
            f"{_pct(jd['claim_supported'])} of {jd['claims_judged']} "
            f"({jd['claim_not_supported']} not supported)",
        ),
        (
            "Judge",
            "Answers correct / correct or partial vs reference",
            f"{_pct(jd['answer_correct'])} / {_pct(jd['answer_correct_or_partial'])} "
            f"of {jd['answers_judged']}",
        ),
        ("Judge", "Questions the judge failed on (excluded above)", str(jd["judge_errors"])),
        ("Cost", "LLM calls", f"{cl['llm_calls']} of {c['total']} questions"),
        (
            "Latency",
            "Server time p50 / p95 (questions that called the LLM)",
            f"{_ms(cl['latency_p50_ms'])} / {_ms(cl['latency_p95_ms'])}",
        ),
        (
            "Tokens",
            "Avg input / output per LLM call",
            f"{_num(cl['avg_input_tokens'], 0)} / {_num(cl['avg_output_tokens'], 0)}",
        ),
    ]


def _problems(results: list[ItemResult]) -> list[str]:
    lines = []
    for r in results:
        if r.error:
            lines.append(f"- `{r.item.id}`: error: {r.error.splitlines()[0]}")
        elif not m.status_ok(r):
            expected = ", ".join(r.item.acceptable_statuses)
            lines.append(f"- `{r.item.id}`: status `{r.status}`, expected one of: {expected}")
        if r.judge_error:
            lines.append(f"- `{r.item.id}`: judge failed (not counted): {r.judge_error}")
        if r.answer_judgment and r.answer_judgment.verdict != "correct":
            lines.append(
                f"- `{r.item.id}`: answer {r.answer_judgment.verdict}: {r.answer_judgment.reason}"
            )
        for j in r.claim_judgments:
            if j.verdict != "supported":
                lines.append(f'- `{r.item.id}`: claim {j.verdict}: "{j.claim}" ({j.reason})')
    return lines


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else math.nan


def _mean(values: list[Any]) -> float:
    numbers = [v for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else math.nan


def _pct(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _ms(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value / 1000:.1f}s"


def _finite(value: Any) -> Any:
    """NaN means 'no data' here; JSON has no NaN, so write null instead."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value
