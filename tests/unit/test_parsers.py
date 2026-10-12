import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from grounded.errors import UnprocessableDocumentError, UnsupportedDocumentError
from grounded.ingestion.parsers import (
    detect_content_type,
    normalize_text,
    parse_document,
    repair_pdf_line_breaks,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _blank_pdf(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.pdf", "application/pdf"),
        ("A.PDF", "application/pdf"),
        ("notes.md", "text/markdown"),
        ("notes.markdown", "text/markdown"),
        ("notes.txt", "text/plain"),
    ],
)
def test_supported_extensions(filename: str, expected: str) -> None:
    assert detect_content_type(filename) == expected


@pytest.mark.parametrize("filename", ["report.docx", "image.png", "noextension"])
def test_unsupported_extensions_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        detect_content_type(filename)


def test_pdf_text_is_extracted_per_page() -> None:
    parsed = parse_document("s.pdf", (FIXTURES / "sample.pdf").read_bytes(), max_pdf_pages=10)

    assert parsed.page_count == 2
    assert [p.number for p in parsed.pages] == [1, 2]
    assert "GOVERN function" in parsed.pages[0].text
    assert "MEASURE function" in parsed.pages[1].text


def test_fake_pdf_is_rejected_by_content_not_extension() -> None:
    with pytest.raises(UnsupportedDocumentError, match="not a PDF"):
        parse_document("evil.pdf", b"MZ\x90\x00 this is an exe", max_pdf_pages=10)


def test_corrupt_pdf_becomes_a_clean_client_error() -> None:
    with pytest.raises(UnprocessableDocumentError, match="could not be read"):
        parse_document("broken.pdf", b"%PDF-1.7\n garbage garbage", max_pdf_pages=10)


def test_pdf_without_text_is_rejected() -> None:
    with pytest.raises(UnprocessableDocumentError, match="No extractable text"):
        parse_document("scan.pdf", _blank_pdf(), max_pdf_pages=10)


def test_pdf_page_limit_is_enforced() -> None:
    with pytest.raises(UnprocessableDocumentError, match="limit is 2"):
        parse_document("big.pdf", _blank_pdf(pages=3), max_pdf_pages=2)


def test_utf8_text_with_bom_is_decoded() -> None:
    parsed = parse_document("n.txt", "﻿Héllo wörld".encode(), max_pdf_pages=10)

    assert parsed.pages[0].text == "Héllo wörld"
    assert parsed.pages[0].number is None
    assert parsed.page_count is None


def test_non_utf8_text_is_rejected() -> None:
    with pytest.raises(UnprocessableDocumentError, match="UTF-8"):
        parse_document("n.txt", "caf\xe9".encode("latin-1"), max_pdf_pages=10)


def test_binary_disguised_as_text_is_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError, match="binary"):
        parse_document("n.txt", b"abc\x00\x01\x02", max_pdf_pages=10)


def test_whitespace_only_text_is_rejected() -> None:
    with pytest.raises(UnprocessableDocumentError):
        parse_document("n.md", b"   \n\n\t  ", max_pdf_pages=10)


def test_markdown_flag() -> None:
    assert parse_document("n.md", b"# Hi", max_pdf_pages=10).is_markdown


def test_normalize_text() -> None:
    raw = "a  b\t\tc   \r\nline2\r\n\r\n\r\n\r\nline3\x00"

    assert normalize_text(raw) == "a b c\nline2\n\nline3"


# --- PDF line-break repair ----------------------------------------------------


def test_soft_hyphen_is_joined_when_the_word_appears_elsewhere() -> None:
    pages = ["Examples are illus-\ntrated below.", "As illustrated in Figure 2."]

    assert repair_pdf_line_breaks(pages)[0] == "Examples are illustrated below."


def test_real_hyphen_is_kept_when_the_compound_appears_elsewhere() -> None:
    pages = ["Use red-\nteaming exercises.", "Plan red-teaming early."]

    assert repair_pdf_line_breaks(pages)[0] == "Use red-teaming exercises."


def test_split_word_is_not_evidence_for_itself() -> None:
    # "subcate-\ngories" must not count "subcategories" as having been seen.
    # No other evidence and no majority: the tie keeps the hyphen.
    assert repair_pdf_line_breaks(["All subcate-\ngories apply."]) == ["All subcate-gories apply."]


def test_ambiguous_split_follows_the_document_majority() -> None:
    soft_hyphen_document = [
        "illus-\ntrated and manag-\ning and subcate-\ngories",
        "illustrated managing",
    ]

    repaired = repair_pdf_line_breaks(soft_hyphen_document)

    assert repaired[0] == "illustrated and managing and subcategories"


def test_ambiguous_split_keeps_the_hyphen_in_a_real_hyphen_document() -> None:
    real_hyphen_document = [
        "red-\nteaming and fine-\ntuning and multi-\nmodal",
        "red-teaming fine-tuning",
    ]

    repaired = repair_pdf_line_breaks(real_hyphen_document)

    assert repaired[0] == "red-teaming and fine-tuning and multi-modal"


@pytest.mark.parametrize(
    "text",
    [
        "the AI-\nRMF document",  # second fragment capitalized
        "risks of-\nthe system",  # stopword fragment
        "years 2023-\n2024 only",  # number ranges
    ],
)
def test_guarded_splits_without_evidence_are_not_guessed(text: str) -> None:
    # The rest of the document votes "join", but these have no evidence of their own.
    soft_hyphen_majority = "illus-\ntrated manag-\ning, illustrated managing"

    assert repair_pdf_line_breaks([text, soft_hyphen_majority])[0] == text


def test_evidence_overrides_the_guards() -> None:
    pages = ["They in-\nclude the GOV-\nERN function.", "This will include GOVERN."]

    assert repair_pdf_line_breaks(pages)[0] == "They include the GOVERN function."


def test_space_before_punctuation_is_removed() -> None:
    assert repair_pdf_line_breaks(["GOVERN , MAP , and MEASURE ."]) == ["GOVERN, MAP, and MEASURE."]


def test_space_before_punctuation_keeps_decimals_and_leading_dots() -> None:
    text = "version 1 .5 and the .NET runtime"

    assert repair_pdf_line_breaks([text]) == [text]


def test_markdown_and_text_are_not_repaired() -> None:
    parsed = parse_document("n.md", b"keep this-\nas is , exactly", max_pdf_pages=10)

    assert parsed.pages[0].text == "keep this-\nas is , exactly"
