"""Citation verification: never trust the model's citations, check them in code.

For each citation the model returns:
  1. The source_id must be one we actually sent.     (catches invented sources)
  2. The quote must actually appear in that source.  (catches invented quotes)
  3. The quote must be long enough to mean something. (catches "the" as a "quote")

Then the answer is adjusted:
  - Markers like [S4] that point at no verified citation are removed.
  - "answered" with some rejected citations becomes "partial".
  - No verified citations at all becomes "insufficient_context": an answer we
    can't back with evidence is not shown.

Honest limit: this proves each quote EXISTS in its source. It does not prove
the answer's wording faithfully reflects the quote. That's measured
separately by the groundedness eval (M7).
"""

import html
import re
import unicodedata
from dataclasses import dataclass, field

from grounded.generation.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    AnswerStatus,
    LLMAnswer,
    Source,
)

MIN_QUOTE_CHARS = 12
_MARKER = re.compile(r"\s*\[(S\d+)\]")
_WHITESPACE = re.compile(r"\s+")
_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize_for_match(text: str) -> str:
    """Compare text the way a human would: ignore case, spacing, and curly-vs-straight quotes."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).translate(_TYPOGRAPHY)
    return _WHITESPACE.sub(" ", text).casefold().strip()


@dataclass(frozen=True)
class VerifiedCitation:
    source: Source
    quote: str


@dataclass(frozen=True)
class RejectedCitation:
    source_id: str
    quote: str
    reason: str  # "unknown_source" | "quote_not_found" | "quote_too_short"


@dataclass(frozen=True)
class VerificationResult:
    status: AnswerStatus
    answer: str
    verified: list[VerifiedCitation]
    rejected: list[RejectedCitation] = field(default_factory=list)
    downgraded_from: AnswerStatus | None = None


def verify_answer(llm_answer: LLMAnswer, sources: list[Source]) -> VerificationResult:
    if llm_answer.status == "insufficient_context":
        # Trust a refusal; drop any stray citations.
        return VerificationResult(
            status="insufficient_context",
            answer=_strip_markers(llm_answer.answer, keep=set()) or INSUFFICIENT_CONTEXT_MESSAGE,
            verified=[],
        )

    by_id = {source.source_id: source for source in sources}
    verified: list[VerifiedCitation] = []
    rejected: list[RejectedCitation] = []
    seen: set[tuple[str, str]] = set()

    for citation in llm_answer.citations:
        quote = citation.quote.strip().strip(".,;:").strip()
        normalized_quote = normalize_for_match(quote)
        key = (citation.source_id, normalized_quote)
        if key in seen:
            continue  # exact duplicate citation: ignore silently
        seen.add(key)

        source = by_id.get(citation.source_id)
        if source is None:
            rejected.append(RejectedCitation(citation.source_id, quote, "unknown_source"))
        elif len(normalized_quote) < MIN_QUOTE_CHARS:
            rejected.append(RejectedCitation(citation.source_id, quote, "quote_too_short"))
        elif normalized_quote not in normalize_for_match(source.chunk.content):
            rejected.append(RejectedCitation(citation.source_id, quote, "quote_not_found"))
        else:
            verified.append(VerifiedCitation(source=source, quote=quote))

    if not verified:
        return VerificationResult(
            status="insufficient_context",
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            verified=[],
            rejected=rejected,
            downgraded_from=llm_answer.status,
        )

    status: AnswerStatus = llm_answer.status
    downgraded_from: AnswerStatus | None = None
    if status == "answered" and rejected:
        status, downgraded_from = "partial", "answered"

    supported_ids = {citation.source.source_id for citation in verified}
    return VerificationResult(
        status=status,
        answer=_strip_markers(llm_answer.answer, keep=supported_ids),
        verified=verified,
        rejected=rejected,
        downgraded_from=downgraded_from,
    )


def _strip_markers(answer: str, keep: set[str]) -> str:
    """Remove [Sx] markers that aren't backed by a verified citation."""
    return _MARKER.sub(lambda m: m.group(0) if m.group(1) in keep else "", answer).strip()
