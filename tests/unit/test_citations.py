"""The claim verifier is the core hallucination defense, so it gets the most tests.

The central guarantee: text from a claim that failed verification can never
appear in what the user sees, no matter how the model formats its response.
"""

import uuid
from typing import Any

import pytest

from grounded.generation.citations import (
    PARTIAL_COVERAGE_NOTICE,
    UNVERIFIED_CLAIMS_NOTICE,
    VerificationResult,
    normalize_for_match,
    verify_answer,
)
from grounded.generation.prompts import INSUFFICIENT_CONTEXT_MESSAGE, LLMAnswer, Source
from grounded.retrieval.retriever import RetrievedChunk

S1_TEXT = "The GOVERN function cultivates a culture of risk management. Leaders are accountable."
S2_TEXT = "The MEASURE function uses quantitative tools to analyze AI risk."

GOVERN_QUOTE = "cultivates a culture of risk management"
MEASURE_QUOTE = "uses quantitative tools to analyze AI risk"


def _source(source_id: str, content: str) -> Source:
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="rmf.pdf",
        chunk_index=0,
        content=content,
        page_start=1,
        page_end=1,
        heading=None,
        score=0.8,
    )
    return Source(source_id=source_id, chunk=chunk)


SOURCES = [_source("S1", S1_TEXT), _source("S2", S2_TEXT)]


def _claim(text: str, *evidence: tuple[str, str]) -> dict[str, Any]:
    return {"text": text, "evidence": [{"source_id": s, "quote": q} for s, q in evidence]}


def _verify(status: str, *claims: dict[str, Any]) -> VerificationResult:
    return verify_answer(
        LLMAnswer.model_validate({"status": status, "claims": list(claims)}), SOURCES
    )


def _visible_text(result: VerificationResult) -> str:
    """Everything a user could read: answer, notice, claims, quotes."""
    parts = [result.answer, result.notice or ""]
    parts += [claim.text for claim in result.claims]
    parts += [evidence.quote for evidence in result.verified]
    return " ".join(parts)


# ----- The core guarantee -----
def test_unsupported_claim_cannot_survive_by_losing_its_citation() -> None:
    """Regression: '...unlimited [S9].' used to become '...unlimited.' and stay visible."""
    result = _verify(
        "answered",
        _claim("Accountability is assigned.", ("S1", "Leaders are accountable")),
        _claim("Budgets are unlimited.", ("S9", "budgets are unlimited for all teams")),
    )

    assert result.answer == "Accountability is assigned [S1]."
    assert "unlimited" not in _visible_text(result).lower()
    assert result.status == "partial"
    assert result.downgraded_from == "answered"
    assert result.notice == UNVERIFIED_CLAIMS_NOTICE


@pytest.mark.parametrize(
    ("bad_claim", "reason"),
    [
        (_claim("ZEBRA budgets are unlimited.", ("S9", "budgets are unlimited")), "unknown_source"),
        (_claim("ZEBRA audits happen weekly.", ("S1", "audits happen weekly")), "quote_not_found"),
        (_claim("ZEBRA is mandatory.", ("S1", "The")), "quote_too_short"),
        (_claim("ZEBRA has no evidence at all."), None),
        (_claim("ZEBRA is in S2.", ("S1", MEASURE_QUOTE)), "quote_not_found"),
        (_claim("ZEBRA cites a malformed id.", ("Source 1", GOVERN_QUOTE)), "unknown_source"),
    ],
)
def test_rejected_claim_text_never_reaches_the_user(
    bad_claim: dict[str, Any], reason: str | None
) -> None:
    result = _verify(
        "answered", _claim("Leaders are accountable.", ("S1", "Leaders are accountable")), bad_claim
    )

    assert "zebra" not in _visible_text(result).lower()
    assert result.answer == "Leaders are accountable [S1]."
    [rejected] = result.rejected_claims
    if reason is None:
        assert rejected.reason == "no_evidence"
    else:
        assert rejected.reason == "invalid_evidence"
        assert [e.reason for e in rejected.evidence] == [reason]


def test_one_fake_quote_rejects_the_whole_claim_even_with_a_real_one() -> None:
    """A model that fabricated support for a claim is not trusted on that claim."""
    result = _verify(
        "answered",
        _claim("GOVERN builds culture.", ("S1", GOVERN_QUOTE)),
        _claim(
            "ZEBRA mixes real and fake evidence.",
            ("S2", MEASURE_QUOTE),
            ("S2", "a sentence that is not in the source"),
        ),
    )

    assert "zebra" not in _visible_text(result).lower()
    assert [c.text for c in result.claims] == ["GOVERN builds culture."]
    assert [e.source.source_id for e in result.verified] == ["S1"], (
        "verified quotes from a rejected claim must not be shown either"
    )


def test_all_claims_rejected_returns_the_fixed_refusal() -> None:
    result = _verify(
        "answered",
        _claim("ZEBRA audits are quarterly.", ("S1", "requires quarterly audits by a firm")),
    )

    assert result.status == "insufficient_context"
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.downgraded_from == "answered"
    assert result.claims == []
    assert result.verified == []
    assert "zebra" not in _visible_text(result).lower()


def test_markers_written_by_the_model_are_replaced_by_verified_ones() -> None:
    """The model claims [S2][S7] in its text, but its only valid evidence is S1."""
    result = _verify(
        "answered", _claim("Leaders are accountable [S2][S7].", ("S1", "Leaders are accountable"))
    )

    assert result.answer == "Leaders are accountable [S1]."
    assert result.claims[0].text == "Leaders are accountable."


def test_model_refusal_shows_the_fixed_message_not_model_text() -> None:
    result = _verify(
        "insufficient_context",
        _claim("ZEBRA: the CEO approves all models.", ("S1", GOVERN_QUOTE)),
    )

    assert result.status == "insufficient_context"
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.claims == []
    assert result.downgraded_from is None


def test_empty_answer_is_insufficient_context() -> None:
    result = _verify("answered")

    assert result.status == "insufficient_context"
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE


def test_blank_claim_text_is_rejected() -> None:
    result = _verify(
        "answered", _claim(" [S1] ", ("S1", GOVERN_QUOTE)), _claim("Real.", ("S2", MEASURE_QUOTE))
    )

    assert [c.reason for c in result.rejected_claims] == ["empty_text"]
    assert result.answer == "Real [S2]."


# ----- Valid answers -----
def test_supported_claims_build_the_answer_with_code_generated_markers() -> None:
    result = _verify(
        "answered",
        _claim("GOVERN builds risk culture.", ("S1", GOVERN_QUOTE)),
        _claim("MEASURE analyzes risk with quantitative tools!", ("S2", MEASURE_QUOTE)),
    )

    assert result.status == "answered"
    assert result.notice is None
    assert result.rejected_claims == []
    assert result.answer == (
        "GOVERN builds risk culture [S1]. MEASURE analyzes risk with quantitative tools [S2]!"
    )


def test_claim_with_several_sources_gets_all_markers_once() -> None:
    result = _verify(
        "answered",
        _claim(
            "Risk is governed and measured",
            ("S1", GOVERN_QUOTE),
            ("S2", MEASURE_QUOTE),
            ("S1", "Leaders are accountable"),
        ),
    )

    assert result.answer == "Risk is governed and measured [S1][S2]."
    assert result.claims[0].source_ids == ["S1", "S2"]
    assert len(result.verified) == 3


def test_model_partial_status_is_kept_with_a_coverage_notice() -> None:
    result = _verify(
        "partial", _claim("Only GOVERN is covered.", ("S1", "Leaders are accountable"))
    )

    assert result.status == "partial"
    assert result.downgraded_from is None
    assert result.notice == PARTIAL_COVERAGE_NOTICE


def test_partial_with_rejected_claims_mentions_both() -> None:
    result = _verify(
        "partial",
        _claim("GOVERN is covered.", ("S1", GOVERN_QUOTE)),
        _claim("ZEBRA is not.", ("S5", GOVERN_QUOTE)),
    )

    assert result.notice == f"{PARTIAL_COVERAGE_NOTICE} {UNVERIFIED_CLAIMS_NOTICE}"


def test_duplicate_claims_and_quotes_are_collapsed() -> None:
    claim = _claim("GOVERN builds culture.", ("S1", GOVERN_QUOTE), ("S1", GOVERN_QUOTE))

    result = _verify("answered", claim, claim)

    assert len(result.claims) == 1
    assert len(result.verified) == 1


# ----- Quote matching -----
def test_matching_ignores_case_spacing_and_curly_quotes() -> None:
    curly = _source("S1", "The policy says “human oversight”   is required.")

    result = verify_answer(
        LLMAnswer.model_validate(
            {
                "status": "answered",
                "claims": [
                    _claim(
                        "Oversight is required.",
                        ("S1", 'THE POLICY SAYS "human oversight" is required.'),
                    )
                ],
            }
        ),
        [curly],
    )

    assert result.status == "answered"


def test_html_escaped_quotes_still_match() -> None:
    """Sources are HTML-escaped in the prompt; the model may quote the escaped form."""
    source = _source("S1", "Risk = likelihood & impact of harm.")

    result = verify_answer(
        LLMAnswer.model_validate(
            {
                "status": "answered",
                "claims": [
                    _claim("Risk has two parts.", ("S1", "Risk = likelihood &amp; impact of harm"))
                ],
            }
        ),
        [source],
    )

    assert result.status == "answered"


def test_normalize_for_match() -> None:
    assert normalize_for_match("  Hello—World\n\t’s  ") == "hello-world 's"


def test_status_capitalization_is_normalized() -> None:
    """Structured outputs may return enum values with different capitalization."""
    answer = LLMAnswer.model_validate({"status": "Answered", "claims": []})

    assert answer.status == "answered"
