"""Provider interfaces.

The rest of the app depends on these Protocols, never on a vendor SDK.
Swapping OpenAI for another embedding provider, or for a fake in tests,
means writing one class that matches the Protocol. Nothing else changes.
"""

from typing import Protocol


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order as the input."""
        ...
