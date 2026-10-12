"""OpenAI embeddings adapter.

Retries: the OpenAI SDK already retries rate limits (429), server errors (5xx)
and dropped connections with exponential backoff (`max_retries`). We rely on
that instead of adding a retry library on top of it.
"""

import time

import openai
import structlog
from openai import AsyncOpenAI

from grounded.errors import EmbeddingProviderError

logger = structlog.get_logger(__name__)


class OpenAIEmbedder:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimension: int,
        batch_size: int,
    ) -> None:
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # Batching: one API call per `batch_size` texts instead of one per chunk.
        # 300 chunks = 3 requests, not 300.
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            response = await self._client.embeddings.create(model=self._model, input=batch)
        except openai.OpenAIError as exc:
            # Log the type and status only; provider messages can echo request details.
            logger.error(
                "embedding_request_failed",
                error_type=type(exc).__name__,
                status=getattr(exc, "status_code", None),
                batch_size=len(batch),
            )
            raise EmbeddingProviderError("The embedding provider failed. Try again later.") from exc

        # The API returns an `index` per item; sort by it rather than trusting order.
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise EmbeddingProviderError(
                f"Embedding provider returned {len(items)} vectors for {len(batch)} inputs"
            )
        for item in items:
            if len(item.embedding) != self._dimension:
                raise EmbeddingProviderError(
                    f"Expected {self._dimension}-dimension vectors, got {len(item.embedding)}"
                )

        logger.info(
            "embedding_batch_completed",
            batch_size=len(batch),
            tokens=response.usage.total_tokens,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return [item.embedding for item in items]
