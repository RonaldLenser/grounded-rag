"""Turn uploaded bytes into clean text, page by page.

Uploaded files are untrusted input. We check the real file contents (magic
bytes), not just the extension, and convert every parser failure into a
clear client error instead of a 500.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

import structlog
from pypdf import PdfReader

from grounded.errors import UnprocessableDocumentError, UnsupportedDocumentError

logger = structlog.get_logger(__name__)

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}
PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based page number for PDFs; None for text/markdown
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    content_type: str
    pages: list[Page]

    @property
    def is_markdown(self) -> bool:
        return self.content_type == "text/markdown"

    @property
    def page_count(self) -> int | None:
        return len(self.pages) if self.content_type == "application/pdf" else None


def detect_content_type(filename: str) -> str:
    extension = PurePath(filename).suffix.lower()
    try:
        return CONTENT_TYPES[extension]
    except KeyError:
        supported = ", ".join(sorted(CONTENT_TYPES))
        raise UnsupportedDocumentError(
            f"Unsupported file type '{extension or '(none)'}'. Supported: {supported}"
        ) from None


def parse_document(filename: str, data: bytes, *, max_pdf_pages: int) -> ParsedDocument:
    """CPU-bound: call from a worker thread (asyncio.to_thread), not the event loop."""
    content_type = detect_content_type(filename)
    if content_type == "application/pdf":
        pages = _parse_pdf(data, max_pdf_pages=max_pdf_pages)
    else:
        pages = [Page(number=None, text=_decode_text(data))]

    pages = [Page(number=page.number, text=normalize_text(page.text)) for page in pages]
    if content_type == "application/pdf":
        repaired = repair_pdf_line_breaks([page.text for page in pages])
        pages = [Page(number=p.number, text=t) for p, t in zip(pages, repaired, strict=True)]
    if not any(page.text for page in pages):
        raise UnprocessableDocumentError(
            "No extractable text found. Scanned/image-only PDFs need OCR, which isn't supported."
        )
    return ParsedDocument(content_type=content_type, pages=pages)


def _parse_pdf(data: bytes, *, max_pdf_pages: int) -> list[Page]:
    if not data.startswith(PDF_MAGIC):
        raise UnsupportedDocumentError("File has a .pdf extension but is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UnprocessableDocumentError("Encrypted PDFs are not supported")
        if len(reader.pages) > max_pdf_pages:
            raise UnprocessableDocumentError(
                f"PDF has {len(reader.pages)} pages; the limit is {max_pdf_pages}"
            )
        return [
            Page(number=index + 1, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages)
        ]
    except UnprocessableDocumentError:
        raise
    except Exception as exc:
        # Malformed PDFs can make the parser raise almost anything. This is the
        # boundary where untrusted input gets converted into a clean 422.
        logger.warning("pdf_parse_failed", error_type=type(exc).__name__)
        raise UnprocessableDocumentError(
            "The PDF could not be read (corrupt or malformed)"
        ) from exc


def _decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise UnsupportedDocumentError("File looks binary, not text")
    try:
        return data.decode("utf-8-sig")  # -sig strips a BOM some Windows editors add
    except UnicodeDecodeError as exc:
        raise UnprocessableDocumentError("Text files must be UTF-8 encoded") from exc


_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")
_THREE_PLUS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Make whitespace predictable so chunking and quote-matching behave consistently."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _TRAILING_SPACE.sub("\n", text)
    text = _RUNS_OF_SPACES.sub(" ", text)
    text = _THREE_PLUS_NEWLINES.sub("\n\n", text)
    return text.strip()


# --- PDF line-break repair ----------------------------------------------------
#
# PDF text extraction keeps the typesetter's line breaks. A word split across
# lines comes out as "illus-\ntrated", which shows up in citation quotes as
# "illus- trated" and hides the word from keyword search. The hard part: a
# trailing hyphen is sometimes a soft hyphen (join: "illustrated") and
# sometimes a real one ("red-\nteaming" must stay "red-teaming").
#
# The text can't tell us which, but the rest of the DOCUMENT usually can:
#   1. "red-teaming" appears inline elsewhere  -> it's a real hyphen: keep it
#   2. "illustrated" appears as a word elsewhere -> soft hyphen: join
#   3. no evidence -> follow this document's majority (typesetters are
#      consistent within one document); ties keep the hyphen, because a wrong
#      hyphen is less damaging than a wrongly glued word.
# Evidence always wins. Without evidence, splits whose fragments are stopwords,
# numbers, or start uppercase ("of-\nthe", "2023-\n2024", "End-\nUsers") are
# left as extracted instead of guessed: the majority rule is unreliable there.

_LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
_WORD = re.compile(r"\w+")
_COMPOUND = re.compile(r"\w+-\w+")
_SPACE_BEFORE_PUNCT = re.compile(r"(?<=\w) +(?=[,.;:!?](?:\s|$))")
_NEVER_JOIN = frozenset(
    {"a", "an", "and", "or", "the", "to", "of", "in", "on", "for", "with", "by"}
)


def repair_pdf_line_breaks(texts: list[str]) -> list[str]:
    """Rejoin words hyphenated across line breaks, using the whole document as evidence.

    Takes every page of one document (evidence is document-wide) and returns the
    pages in the same order. Also removes stray spaces before punctuation that
    PDF extraction inserts ("GOVERN , MAP" -> "GOVERN, MAP").
    """
    # Evidence comes from text OUTSIDE the line-break hyphens, so a split word
    # can't count as evidence for itself.
    flat = _LINE_BREAK_HYPHEN.sub(" ", "\n".join(texts)).lower()
    words = set(_WORD.findall(flat))
    compounds = set(_COMPOUND.findall(flat))

    decisions: dict[tuple[str, str], str | None] = {}  # "keep" | "join" | None (unknown)
    votes = {"keep": 0, "join": 0}
    for match in _LINE_BREAK_HYPHEN.finditer("\n".join(texts)):
        first, second = match.group(1), match.group(2)
        decision = _evidence(first, second, words, compounds)
        decisions[(first, second)] = decision
        if decision is not None:
            votes[decision] += 1
    default = "join" if votes["join"] > votes["keep"] else "keep"

    def fix(match: re.Match[str]) -> str:
        first, second = match.group(1), match.group(2)
        decision = decisions.get((first, second))
        if decision is None:
            if _is_guarded(first, second):
                return match.group(0)  # no evidence and not a safe guess: leave as extracted
            decision = default
        return first + second if decision == "join" else f"{first}-{second}"

    repaired = [_LINE_BREAK_HYPHEN.sub(fix, text) for text in texts]
    return [_SPACE_BEFORE_PUNCT.sub("", text) for text in repaired]


def _is_guarded(first: str, second: str) -> bool:
    return (
        second[0].isupper()
        or first.lower() in _NEVER_JOIN
        or second.lower() in _NEVER_JOIN
        or first.isdigit()
        or second.isdigit()
    )


def _evidence(first: str, second: str, words: set[str], compounds: set[str]) -> str | None:
    if f"{first}-{second}".lower() in compounds:
        return "keep"
    if (first + second).lower() in words:
        return "join"
    return None
