"""Test doubles for external providers. No network, no API keys, no cost."""

import hashlib
import html
import math
import re
from typing import Any

from grounded.db.models import EMBEDDING_DIM
from grounded.errors import EmbeddingProviderError, LLMProviderError
from grounded.providers.base import OutputSchema, StructuredCompletion

_WORD = re.compile(r"[a-z0-9]+")
_SOURCE_BLOCK = re.compile(r'<source id="S1"[^>]*>\n(?P<content>.*?)\n</source>', re.DOTALL)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    ]
)


class FakeEmbedder:
    """Deterministic, offline embedder with crude "meaning": shared words = similar vectors.

    Technique: feature hashing (bag of words). Each word is hashed to one of the
    1536 dimensions; a text's vector counts its words there, then is normalized.
    Two texts sharing words point in similar directions, so cosine similarity
    behaves sensibly enough to test that retrieval ranks the right chunk first.

    It is NOT semantic: "car" and "automobile" share nothing here. Real meaning
    comes from the real model; this only makes retrieval logic testable.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-embedding-model"

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingProviderError("The embedding provider failed. Try again later.")
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text: str) -> list[float]:
        values = [0.0] * EMBEDDING_DIM
        values[0] = 0.01  # never a zero vector (cosine similarity is undefined for zero)
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            values[index] += 1.0
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]


class FakeLLM:
    """Offline stand-in for Claude.

    By default it behaves like a well-behaved model: it reads the sources in the
    prompt, answers from S1, and quotes S1's first sentence exactly. Tests can
    instead script an exact response (`respond_with`) to simulate bad behavior:
    invented sources, invented quotes, refusals, malformed JSON.
    """

    def __init__(self) -> None:
        self.respond_with: dict[str, Any] | None = None
        self.fail = False
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return "fake-llm"

    async def complete_structured(
        self, *, system: str, user: str, output: OutputSchema, max_tokens: int
    ) -> StructuredCompletion:
        self.calls.append({"system": system, "user": user, "schema": output.name})
        if self.fail:
            raise LLMProviderError("The language model failed. Try again later.")
        data = self.respond_with if self.respond_with is not None else self._grounded(user)
        return StructuredCompletion(data=data, model=self.model, input_tokens=100, output_tokens=20)

    @staticmethod
    def _grounded(user: str) -> dict[str, Any]:
        match = _SOURCE_BLOCK.search(user)
        if match is None:
            return {"status": "insufficient_context", "answer": "No sources.", "citations": []}
        content = html.unescape(match.group("content"))
        first_sentence = content.split(". ")[0].split("\n")[-1].strip()
        return {
            "status": "answered",
            "answer": f"According to the documents, {first_sentence} [S1]",
            "citations": [{"source_id": "S1", "quote": first_sentence}],
        }
