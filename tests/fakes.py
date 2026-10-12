"""Test doubles for external providers. No network, no API keys, no cost."""

import hashlib
import math

from grounded.db.models import EMBEDDING_DIM
from grounded.errors import EmbeddingProviderError


class FakeEmbedder:
    """Deterministic embedder: the same text always gives the same unit vector.

    Vectors come from a hash of the text, so they carry no meaning, but they're
    stable, the right size, and let tests check exactly what was embedded.
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
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        raw = [seed[i % len(seed)] - 127.5 + i % 7 for i in range(EMBEDDING_DIM)]
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]
