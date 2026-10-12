"""Contract tests for the Anthropic adapter: the real SDK against a fake HTTP transport."""

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic

from evals import judge
from grounded.errors import LLMProviderError
from grounded.generation.prompts import ANSWER_SCHEMA
from grounded.providers.anthropic_llm import AnthropicLLM
from grounded.providers.base import OutputSchema

SCHEMA = OutputSchema(
    name="t",
    schema={
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    },
)


def _llm(handler: Callable[[httpx2.Request], httpx2.Response]) -> AnthropicLLM:
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicLLM(client=client, model="claude-sonnet-5-5")


def _message(text: str | None, stop_reason: str = "end_turn") -> httpx2.Response:
    content = [] if text is None else [{"type": "text", "text": text}]
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 30},
        },
    )


async def _call(llm: AnthropicLLM) -> Any:
    return await llm.complete_structured(system="sys", user="usr", output=SCHEMA, max_tokens=256)


async def test_sends_json_schema_and_parses_the_reply() -> None:
    sent: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.update(json.loads(request.content))
        return _message('{"status": "ok"}')

    result = await _call(_llm(handler))

    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA.schema}}
    assert "tool_choice" not in sent, "forced tool use returns 400 on Sonnet 5.5"
    assert sent["system"] == "sys"
    assert sent["max_tokens"] == 256
    assert result.data == {"status": "ok"}
    assert (result.input_tokens, result.output_tokens) == (120, 30)


async def test_refusal_is_an_error() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return _message("I can't help with that.", stop_reason="refusal")

    with pytest.raises(LLMProviderError, match="declined"):
        await _call(_llm(handler))


async def test_truncated_output_is_an_error() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return _message('{"status": "o', stop_reason="max_tokens")

    with pytest.raises(LLMProviderError, match="cut off"):
        await _call(_llm(handler))


@pytest.mark.parametrize("text", ["not json at all", "[1, 2, 3]", None])
async def test_non_object_output_is_an_error(text: str | None) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return _message(text)

    with pytest.raises(LLMProviderError, match="no usable answer"):
        await _call(_llm(handler))


@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
async def test_api_errors_become_llm_provider_errors(status: int) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"type": "error", "error": {"type": "x", "message": "nope"}}
        )

    with pytest.raises(LLMProviderError):
        await _call(_llm(handler))


UNSUPPORTED_KEYWORDS = {"minLength", "maxLength", "minimum", "maximum", "multipleOf", "maxItems"}


def _walk(node: Any, path: str = "$") -> list[str]:
    problems: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            problems.append(f"{path}: objects need additionalProperties: false")
        problems += [f"{path}: unsupported '{k}'" for k in UNSUPPORTED_KEYWORDS & node.keys()]
        for key, value in node.items():
            problems += _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            problems += _walk(value, f"{path}[{index}]")
    return problems


@pytest.mark.parametrize(
    "schema",
    [ANSWER_SCHEMA, judge.CLAIM_SCHEMA, judge.ANSWER_SCHEMA],
    ids=lambda schema: schema.name,
)
def test_schemas_follow_structured_output_rules(schema: OutputSchema) -> None:
    """Regression test: a schema the API rejects fails here, not as a 400 in production."""
    assert _walk(schema.schema) == []
