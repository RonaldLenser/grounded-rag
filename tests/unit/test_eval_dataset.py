"""The golden set is data the metrics depend on, so it's validated like code."""

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.dataset import DEFAULT_GOLDEN_PATH, GoldenItem, load_golden

SEED_SCRIPT = Path(__file__).parents[2] / "scripts" / "seed_demo.py"


def _item(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q-1",
        "category": "answerable",
        "question": "What is GOVERN?",
        "expected_sources": [{"file": "a.pdf", "pages": [3]}],
        "reference_answer": "A function.",
        "acceptable_statuses": ["answered", "partial"],
    }
    return {**base, **overrides}


def test_golden_set_loads_and_covers_every_category() -> None:
    items = load_golden()

    counts = Counter(item.category for item in items)
    assert len(items) >= 30
    assert counts["answerable"] >= 15
    assert counts["unanswerable"] >= 5
    assert counts["out_of_domain"] >= 2
    assert counts["multi_source"] >= 2


def test_golden_sources_name_the_documents_the_demo_seeds() -> None:
    """A typo in a filename would silently score every retrieval as a miss."""
    seeded = set(re.findall(r'"([\w.-]+\.pdf)"', SEED_SCRIPT.read_text()))

    files = {s.file for item in load_golden() for s in item.expected_sources}

    assert seeded
    assert files <= seeded


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    line = DEFAULT_GOLDEN_PATH.read_text().splitlines()[0]
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{line}\n{line}\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_sources": []},  # answerable without sources
        {"reference_answer": None},  # answerable without a reference
        {"acceptable_statuses": ["insufficient_context"]},  # answerable that may refuse
        {"category": "unanswerable", "reference_answer": None},  # unanswerable with sources
        {
            "category": "unanswerable",
            "expected_sources": [],
            "reference_answer": None,
            "acceptable_statuses": ["answered"],
        },
        {"category": "multi_source"},  # "multi" source from one document
    ],
)
def test_inconsistent_items_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GoldenItem.model_validate(_item(**overrides))
