"""Split document text into overlapping, token-limited chunks.

How it works:
1. Break each page into paragraphs (blank lines), then sentences.
   A sentence longer than the chunk size is hard-split by tokens (last resort).
2. Pack sentences into a chunk until the next one would exceed chunk_size.
3. Start the next chunk with the last few sentences of the previous one
   (up to chunk_overlap tokens), so a fact on a boundary appears whole somewhere.
4. In Markdown, a new heading always starts a new chunk, so one chunk never
   mixes two sections.

Every chunk keeps the page range it came from: that's what citations show.

Why tokens, not characters: embedding models and LLMs have token limits,
and "500 characters" means very different amounts of meaning across texts.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from grounded.ingestion.parsers import Page

# cl100k_base is the tokenizer used by OpenAI's text-embedding-3 models.
ENCODING_NAME = "cl100k_base"

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_SEPARATOR_TOKENS = 1  # cost of the space / blank line we insert between pieces


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    heading: str | None


@dataclass(frozen=True)
class _Piece:
    """A sentence (or part of one) plus where it came from."""

    text: str
    tokens: int
    page: int | None
    heading: str | None
    starts_paragraph: bool


def chunk_pages(
    pages: list[Page],
    *,
    chunk_size: int,
    chunk_overlap: int,
    is_markdown: bool,
) -> list[Chunk]:
    """CPU-bound: call from a worker thread (asyncio.to_thread), not the event loop."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    pieces = _split_into_pieces(pages, chunk_size=chunk_size, is_markdown=is_markdown)
    return _pack(pieces, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _split_into_pieces(
    pages: list[Page], *, chunk_size: int, is_markdown: bool
) -> Iterator[_Piece]:
    heading: str | None = None
    for page in pages:
        for paragraph in _PARAGRAPH_BREAK.split(page.text):
            lines = [line.strip() for line in paragraph.strip().split("\n") if line.strip()]
            if not lines:
                continue

            if is_markdown and (match := _MARKDOWN_HEADING.match(lines[0])):
                heading = match.group("title")
                # The heading line itself stays in the text: it carries meaning for search.
                yield _make_piece(lines[0], page.number, heading, starts_paragraph=True)
                lines = lines[1:]
                if not lines:
                    continue

            # Line breaks inside a paragraph are usually layout (PDF columns, wrapping).
            text = " ".join(lines)
            for position, sentence in enumerate(_SENTENCE_END.split(text)):
                for part_number, part in enumerate(_hard_split(sentence, chunk_size)):
                    yield _make_piece(
                        part,
                        page.number,
                        heading,
                        starts_paragraph=position == 0 and part_number == 0,
                    )


def _make_piece(
    text: str, page: int | None, heading: str | None, *, starts_paragraph: bool
) -> _Piece:
    return _Piece(
        text=text,
        tokens=count_tokens(text),
        page=page,
        heading=heading,
        starts_paragraph=starts_paragraph,
    )


def _hard_split(sentence: str, max_tokens: int) -> list[str]:
    """Last resort for a 'sentence' longer than a whole chunk (tables, code, no punctuation)."""
    encoding = get_encoding()
    tokens = encoding.encode(sentence)
    if len(tokens) <= max_tokens:
        return [sentence]
    return [
        encoding.decode(tokens[start : start + max_tokens]).strip()
        for start in range(0, len(tokens), max_tokens)
    ]


def _pack(pieces: Iterator[_Piece], *, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[_Piece] = []

    for piece in pieces:
        if current:
            heading_changed = piece.heading != current[-1].heading
            too_big = _size(current) + _SEPARATOR_TOKENS + piece.tokens > chunk_size
            if heading_changed or too_big:
                chunks.append(_build_chunk(len(chunks), current))
                # New section: no overlap (it would drag text from the old section in).
                current = [] if heading_changed else _overlap_tail(current, chunk_overlap)
                if current and _size(current) + _SEPARATOR_TOKENS + piece.tokens > chunk_size:
                    current = []  # overlap + this piece won't fit: start clean
        current.append(piece)

    if current:
        chunks.append(_build_chunk(len(chunks), current))
    return chunks


def _size(pieces: list[_Piece]) -> int:
    return sum(p.tokens for p in pieces) + _SEPARATOR_TOKENS * (len(pieces) - 1)


def _overlap_tail(pieces: list[_Piece], chunk_overlap: int) -> list[_Piece]:
    """The last whole sentences of a chunk that fit inside the overlap budget."""
    tail: list[_Piece] = []
    for piece in reversed(pieces):
        if _size([piece, *tail]) > chunk_overlap:
            break
        tail.insert(0, piece)
    # Never repeat an entire chunk.
    return tail[1:] if len(tail) == len(pieces) else tail


def _build_chunk(index: int, pieces: list[_Piece]) -> Chunk:
    parts = [pieces[0].text]
    for piece in pieces[1:]:
        parts.append(("\n\n" if piece.starts_paragraph else " ") + piece.text)
    text = "".join(parts)

    pages = [p.page for p in pieces if p.page is not None]
    return Chunk(
        index=index,
        text=text,
        token_count=count_tokens(text),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        heading=pieces[-1].heading,
    )
