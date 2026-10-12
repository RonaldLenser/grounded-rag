"""What one evaluated question produced: raw API data plus judge verdicts.

Kept as plain Pydantic models so a whole run serializes to JSON and can be
re-scored or audited later without calling any API again.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from evals.dataset import GoldenItem

ClaimVerdict = Literal["supported", "partially_supported", "not_supported"]
AnswerVerdict = Literal["correct", "partially_correct", "incorrect"]


class SearchHit(BaseModel):
    filename: str
    page_start: int | None
    page_end: int | None
    score: float


class ClaimJudgment(BaseModel):
    claim: str
    verdict: ClaimVerdict
    reason: str


class AnswerJudgment(BaseModel):
    verdict: AnswerVerdict
    reason: str


class ItemResult(BaseModel):
    item: GoldenItem
    search_hits: list[SearchHit] = Field(default_factory=list)
    response: dict[str, Any] | None = None  # the /query response body, as returned
    error: str | None = None
    claim_judgments: list[ClaimJudgment] = Field(default_factory=list)
    answer_judgment: AnswerJudgment | None = None

    @property
    def status(self) -> str | None:
        return None if self.response is None else str(self.response["status"])

    @property
    def top_score(self) -> float | None:
        return self.search_hits[0].score if self.search_hits else None
