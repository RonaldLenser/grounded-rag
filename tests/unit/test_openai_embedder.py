"""Contract tests for the OpenAI adapter.

The real SDK runs against a fake HTTP transport, so these tests exercise our
actual request/response handling (batching, ordering, validation, errors)
without the network.
"""

import json
from collections.abc import Callable

import httpx2
import pytest
from openai import AsyncOpenAI

from grounded.errors import EmbeddingProviderError
from grounded.providers.openai_embedder import OpenAIEmbedder

DIM = 4


def _embedder(
    handler: Callable[[httpx2.Request], httpx2.Response], batch_size: int = 2
) -> OpenAIEmbedder:
    client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return OpenAIEmbedder(
        client=client, model="text-embedding-3-small", dimension=DIM, batch_size=batch_size
    )


def _ok_response(inputs: list[str], *, reverse: bool = False, dim: int = DIM) -> httpx2.Response:
    data = [
        {"object": "embedding", "index": i, "embedding": [float(len(text))] * dim}
        for i, text in enumerate(inputs)
    ]
    if reverse:
        data.reverse()
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        },
    )


async def test_texts_are_sent_in_batches() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        inputs = json.loads(request.content)["input"]
        requests.append(inputs)
        return _ok_response(inputs)

    vectors = await _embedder(handler, batch_size=2).embed(["a", "bb", "ccc", "dddd", "e"])

    assert requests == [["a", "bb"], ["ccc", "dddd"], ["e"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 1.0]


async def test_results_are_reordered_by_index() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _ok_response(json.loads(request.content)["input"], reverse=True)

    vectors = await _embedder(handler).embed(["a", "bbb"])

    assert [v[0] for v in vectors] == [1.0, 3.0]


async def test_wrong_dimension_is_rejected() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _ok_response(json.loads(request.content)["input"], dim=DIM + 1)

    with pytest.raises(EmbeddingProviderError, match="dimension"):
        await _embedder(handler).embed(["a"])


@pytest.mark.parametrize("status", [400, 401, 429, 500])
async def test_api_errors_become_embedding_provider_errors(status: int) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "nope", "type": "x"}})

    with pytest.raises(EmbeddingProviderError):
        await _embedder(handler).embed(["a"])


async def test_empty_input_makes_no_requests() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        raise AssertionError("should not be called")

    assert await _embedder(handler).embed([]) == []
