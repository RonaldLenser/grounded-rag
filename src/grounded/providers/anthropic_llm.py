"""Anthropic (Claude) adapter using native structured outputs.

Structured outputs: we send a JSON Schema in `output_config.format`, and the
API constrains Claude's reply to match it. No "please respond in JSON"
prompting, no regex-parsing JSON out of prose.

History (a real debugging story): this adapter first used FORCED TOOL USE
(tool_choice={"type": "tool"}) to get structured output. Claude Sonnet 5.5
rejects forced tool use with a 400, so we moved to structured outputs, the
provider's dedicated feature for "reply in exactly this shape."

We still validate the parsed JSON with Pydantic afterward (the caller does):
a schema-shaped reply can still contain values we must reject.

Retries for 429/5xx/connection errors come from the SDK (`max_retries`).
"""

import json
import time

import anthropic
import structlog
from anthropic import AsyncAnthropic

from grounded.errors import LLMProviderError
from grounded.providers.base import OutputSchema, StructuredCompletion

logger = structlog.get_logger(__name__)


class AnthropicLLM:
    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete_structured(
        self, *, system: str, user: str, output: OutputSchema, max_tokens: int
    ) -> StructuredCompletion:
        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": output.schema}},
            )
        except anthropic.APIStatusError as exc:
            # 4xx means OUR request was wrong; the provider's message says why
            # (e.g. an unsupported parameter). Log it, truncated. 5xx: type only.
            logger.error(
                "llm_request_failed",
                error_type=type(exc).__name__,
                status=exc.status_code,
                provider_message=_provider_message(exc) if exc.status_code < 500 else None,
            )
            raise LLMProviderError("The language model failed. Try again later.") from exc
        except anthropic.AnthropicError as exc:
            logger.error("llm_request_failed", error_type=type(exc).__name__)
            raise LLMProviderError("The language model failed. Try again later.") from exc

        if response.stop_reason == "refusal":
            logger.warning("llm_refused", model=self._model)
            raise LLMProviderError("The language model declined to answer this question.")
        if response.stop_reason == "max_tokens":
            # The JSON was cut off mid-object: unusable. Fix: raise LLM_MAX_TOKENS.
            logger.error("llm_output_truncated", max_tokens=max_tokens)
            raise LLMProviderError("The language model's answer was too long and got cut off.")

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("llm_output_not_json", stop_reason=response.stop_reason)
            raise LLMProviderError("The language model returned no usable answer.") from exc
        if not isinstance(data, dict):
            raise LLMProviderError("The language model returned no usable answer.")

        logger.info(
            "llm_completed",
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return StructuredCompletion(
            data=data,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _provider_message(exc: anthropic.APIStatusError) -> str:
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str):
            return message[:300]
    return str(exc)[:300]
