"""Provider interfaces.

The rest of the app depends on these Protocols, never on a vendor SDK.
Swapping a provider, or using a fake in tests, means writing one class that
matches the Protocol. Nothing else changes.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order as the input."""
        ...


@dataclass(frozen=True)
class OutputSchema:
    """The exact JSON shape the model's reply must have (a JSON Schema)."""

    name: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class StructuredCompletion:
    data: dict[str, Any]  # parsed JSON matching the schema; still validated by the caller
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    async def complete_structured(
        self, *, system: str, user: str, output: OutputSchema, max_tokens: int
    ) -> StructuredCompletion:
        """Get a reply constrained to `output.schema`, parsed into a dict."""
        ...
