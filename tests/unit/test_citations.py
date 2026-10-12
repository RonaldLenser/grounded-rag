"""The citation verifier is the core hallucination defense, so it gets the most tests."""

import uuid

from grounded.generation.citations import normalize_for_match, verify_answer
from grounded.generation.prompts import INSUFFICIENT_CONTEXT_MESSAGE, LLMAnswer, Source
from grounded.retrieval.retriever import RetrievedChunk

S1_TEXT = "The GOVERN function cultivates a culture of risk management. Leaders are accountable."
S2_TEXT = "The MEASURE function uses quantitative tools to analyze AI risk."


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


def _answer(status: str, answer: str, citations: list[tuple[str, str]]) -> LLMAnswer:
    return LLMAnswer.model_validate(
        {
            "status": status,
            "answer": answer,
            "citations": [{"source_id": s, "quote": q} for s, q in citations],
        }
    )


def test_valid_citations_are_kept() -> None:
    result = verify_answer(
        _answer(
            "answered",
            "GOVERN builds risk culture [S1] and MEASURE analyzes risk [S2].",
            [("S1", "cultivates a culture of risk management"), ("S2", "quantitative tools")],
        ),
        SOURCES,
    )

    assert result.status == "answered"
    assert [c.source.source_id for c in result.verified] == ["S1", "S2"]
    assert result.rejected == []
    assert result.answer == "GOVERN builds risk culture [S1] and MEASURE analyzes risk [S2]."


def test_invented_source_is_rejected_and_its_marker_removed() -> None:
    result = verify_answer(
        _answer(
            "answered",
            "GOVERN builds culture [S1]. Also something else [S7].",
            [("S1", "cultivates a culture of risk management"), ("S7", "made up quote text")],
        ),
        SOURCES,
    )

    assert result.status == "partial"
    assert result.downgraded_from == "answered"
    assert [r.reason for r in result.rejected] == ["unknown_source"]
    assert "[S7]" not in result.answer
    assert "[S1]" in result.answer


def test_invented_quote_is_rejected() -> None:
    result = verify_answer(
        _answer(
            "answered",
            "GOVERN requires quarterly audits [S1].",
            [("S1", "requires quarterly audits by an external firm")],
        ),
        SOURCES,
    )

    assert result.status == "insufficient_context"
    assert result.downgraded_from == "answered"
    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert [r.reason for r in result.rejected] == ["quote_not_found"]


def test_quote_from_the_wrong_source_is_rejected() -> None:
    """The quote exists, but in S2, not S1. A citation must point at the right source."""
    result = verify_answer(
        _answer("answered", "x [S1]", [("S1", "uses quantitative tools to analyze AI risk")]),
        SOURCES,
    )

    assert result.status == "insufficient_context"
    assert result.rejected[0].reason == "quote_not_found"


def test_trivially_short_quotes_are_rejected() -> None:
    result = verify_answer(_answer("answered", "x [S1]", [("S1", "The")]), SOURCES)

    assert result.rejected[0].reason == "quote_too_short"
    assert result.status == "insufficient_context"


def test_matching_ignores_case_spacing_and_curly_quotes() -> None:
    curly = _source("S1", "The policy says “human oversight”   is required.")

    result = verify_answer(
        _answer("answered", "x [S1]", [("S1", 'THE POLICY SAYS "human oversight" is required.')]),
        [curly],
    )

    assert result.status == "answered"


def test_model_refusal_is_trusted_and_stray_citations_dropped() -> None:
    result = verify_answer(
        _answer("insufficient_context", "Not covered [S1].", [("S1", "cultivates a culture")]),
        SOURCES,
    )

    assert result.status == "insufficient_context"
    assert result.verified == []
    assert result.answer == "Not covered."


def test_partial_status_is_preserved_when_citations_verify() -> None:
    result = verify_answer(
        _answer("partial", "Only GOVERN is covered [S1].", [("S1", "Leaders are accountable")]),
        SOURCES,
    )

    assert result.status == "partial"
    assert result.downgraded_from is None


def test_duplicate_citations_are_collapsed() -> None:
    quote = "cultivates a culture of risk management"

    result = verify_answer(_answer("answered", "x [S1]", [("S1", quote), ("S1", quote)]), SOURCES)

    assert len(result.verified) == 1


def test_html_escaped_quotes_still_match() -> None:
    """Sources are HTML-escaped in the prompt; the model may quote the escaped form."""
    source = _source("S1", "Risk = likelihood & impact of harm.")

    result = verify_answer(
        _answer("answered", "x [S1]", [("S1", "Risk = likelihood &amp; impact of harm")]),
        [source],
    )

    assert result.status == "answered"


def test_normalize_for_match() -> None:
    assert normalize_for_match("  Hello—World\n\t’s  ") == "hello-world 's"


def test_status_capitalization_is_normalized() -> None:
    """Structured outputs may return enum values with different capitalization."""
    answer = LLMAnswer.model_validate({"status": "Answered", "answer": "x", "citations": []})

    assert answer.status == "answered"
