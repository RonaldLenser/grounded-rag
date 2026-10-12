"""scripts/seed_demo.py against the real app: plain re-runs skip, --replace re-ingests."""

import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import seed_demo  # noqa: E402

PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


@pytest.fixture
def seed(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(seed_demo, "DOCUMENTS", {"sample.pdf": "https://example.invalid"})
    monkeypatch.setattr(seed_demo, "download", lambda filename, url: PDF.read_bytes())
    monkeypatch.setattr(seed_demo, "load_admin_key", lambda: "unused")
    monkeypatch.setattr(httpx2, "Client", lambda **kwargs: api)
    return seed_demo.main


def _documents(api: TestClient) -> list[dict[str, Any]]:
    [collection] = api.get("/v1/collections").json()
    documents: list[dict[str, Any]] = api.get(
        f"/v1/collections/{collection['id']}/documents"
    ).json()
    return documents


def test_rerun_skips_duplicates_and_replace_reingests(api: TestClient, seed: Any) -> None:
    assert seed("http://testserver") == 0
    [first] = _documents(api)

    assert seed("http://testserver") == 0
    assert [doc["id"] for doc in _documents(api)] == [first["id"]]  # skipped, not re-added

    assert seed("http://testserver", replace=True) == 0
    [replaced] = _documents(api)
    assert replaced["id"] != first["id"]  # deleted and uploaded again
    assert replaced["filename"] == "sample.pdf"
