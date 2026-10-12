"""The golden set: hand-written questions with known answers and known sources.

Each item says what a correct system should do:
  answerable     the answer is in one document; expected pages are known
  multi_source   the answer needs both documents
  unanswerable   plausible, on-topic, but NOT in the documents (verified by text search)
  out_of_domain  unrelated to the documents; the relevance gate should refuse for free

Page numbers are PDF page indexes (1-based), the same numbers the app stores
for chunks, not the printed page labels.
"""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

Category = Literal["answerable", "multi_source", "unanswerable", "out_of_domain"]
Status = Literal["answered", "partial", "insufficient_context"]

ANSWER_CATEGORIES: tuple[Category, ...] = ("answerable", "multi_source")
REFUSE_CATEGORIES: tuple[Category, ...] = ("unanswerable", "out_of_domain")

DEFAULT_GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"


class ExpectedSource(BaseModel):
    file: str
    pages: list[int] = Field(min_length=1)


class GoldenItem(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: Category
    question: str = Field(min_length=1, max_length=1000)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    reference_answer: str | None = None
    acceptable_statuses: list[Status] = Field(min_length=1)

    @property
    def should_answer(self) -> bool:
        return self.category in ANSWER_CATEGORIES

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.should_answer:
            if not self.expected_sources or not self.reference_answer:
                raise ValueError(f"{self.id}: needs expected_sources and reference_answer")
            if "insufficient_context" in self.acceptable_statuses:
                raise ValueError(f"{self.id}: an answerable question can't accept a refusal")
        else:
            if self.expected_sources:
                raise ValueError(f"{self.id}: unanswerable items have no expected sources")
            if "answered" in self.acceptable_statuses:
                raise ValueError(f"{self.id}: an unanswerable question can't accept 'answered'")
        if self.category == "multi_source" and len({s.file for s in self.expected_sources}) < 2:
            raise ValueError(f"{self.id}: multi_source needs sources from 2+ documents")
        return self


def load_golden(path: Path = DEFAULT_GOLDEN_PATH) -> list[GoldenItem]:
    items = [
        GoldenItem.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate golden ids: {sorted(duplicates)}")
    return items
