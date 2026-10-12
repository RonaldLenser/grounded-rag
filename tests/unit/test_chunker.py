import pytest

from grounded.ingestion.chunker import chunk_pages, count_tokens
from grounded.ingestion.parsers import Page


def _sentences(count: int, words: int = 12, prefix: str = "Sentence") -> str:
    """`count` sentences; with the default 12 words each is ~52 tokens."""
    return " ".join(
        f"{prefix} {n} " + " ".join(f"word{n}x{w}" for w in range(words)) + "."
        for n in range(count)
    )


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_pages([Page(1, "")], chunk_size=100, chunk_overlap=10, is_markdown=False) == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_pages(
        [Page(None, "Hello world. Short doc.")], chunk_size=100, chunk_overlap=10, is_markdown=False
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world. Short doc."
    assert chunks[0].index == 0
    assert chunks[0].token_count == count_tokens("Hello world. Short doc.")


def test_no_chunk_exceeds_the_token_limit() -> None:
    pages = [Page(1, _sentences(80))]

    chunks = chunk_pages(pages, chunk_size=120, chunk_overlap=20, is_markdown=False)

    assert len(chunks) > 5
    assert all(chunk.token_count <= 120 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    # Short sentences (~12 tokens) so several fit inside the 40-token overlap budget.
    text = _sentences(60, words=4)

    chunks = chunk_pages([Page(1, text)], chunk_size=150, chunk_overlap=40, is_markdown=False)

    assert len(chunks) > 3
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text, "next chunk should start with the previous tail"


def test_overlap_uses_whole_sentences_only() -> None:
    """Documented tradeoff: overlap never cuts a sentence in half. If even one
    sentence is bigger than the overlap budget, that boundary gets no overlap."""
    text = _sentences(20, words=12)  # each sentence is ~52 tokens > 40-token budget

    chunks = chunk_pages([Page(1, text)], chunk_size=150, chunk_overlap=40, is_markdown=False)

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.text.split(". ")[0] not in previous.text


def test_zero_overlap_means_no_repeated_text() -> None:
    chunks = chunk_pages(
        [Page(1, _sentences(40))], chunk_size=100, chunk_overlap=0, is_markdown=False
    )

    joined = " ".join(chunk.text for chunk in chunks)
    assert joined.count("Sentence 5 ") == 1


def test_chunk_indices_are_sequential() -> None:
    chunks = chunk_pages(
        [Page(1, _sentences(50))], chunk_size=100, chunk_overlap=20, is_markdown=False
    )

    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_page_ranges_follow_the_text() -> None:
    pages = [Page(1, _sentences(10, prefix="One")), Page(2, _sentences(10, prefix="Two"))]

    chunks = chunk_pages(pages, chunk_size=4000, chunk_overlap=0, is_markdown=False)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunk_on_a_single_page_reports_that_page() -> None:
    pages = [Page(7, _sentences(3))]

    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=0, is_markdown=False)

    assert (chunks[0].page_start, chunks[0].page_end) == (7, 7)


def test_text_files_have_no_page_numbers() -> None:
    chunks = chunk_pages(
        [Page(None, "Plain text.")], chunk_size=100, chunk_overlap=0, is_markdown=False
    )

    assert (chunks[0].page_start, chunks[0].page_end) == (None, None)


def test_markdown_headings_start_new_chunks_and_are_recorded() -> None:
    text = "# Intro\n\nWelcome here.\n\n## Setup\n\nInstall it. Run it.\n\n## Usage\n\nUse it."

    chunks = chunk_pages([Page(None, text)], chunk_size=500, chunk_overlap=50, is_markdown=True)

    assert [c.heading for c in chunks] == ["Intro", "Setup", "Usage"]
    assert chunks[1].text == "## Setup\n\nInstall it. Run it."
    assert "Welcome" not in chunks[1].text, "no overlap across sections"


def test_hash_lines_are_not_headings_in_plain_text() -> None:
    chunks = chunk_pages(
        [Page(None, "# not a heading\n\nBody.")], chunk_size=100, chunk_overlap=0, is_markdown=False
    )

    assert chunks[0].heading is None


def test_paragraph_breaks_are_preserved_inside_a_chunk() -> None:
    chunks = chunk_pages(
        [Page(1, "First paragraph.\n\nSecond paragraph.")],
        chunk_size=100,
        chunk_overlap=0,
        is_markdown=False,
    )

    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_oversized_sentence_is_hard_split() -> None:
    giant = " ".join(f"token{n}" for n in range(600))  # no punctuation at all

    chunks = chunk_pages([Page(1, giant)], chunk_size=100, chunk_overlap=0, is_markdown=False)

    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller"):
        chunk_pages([Page(1, "x")], chunk_size=100, chunk_overlap=100, is_markdown=False)
