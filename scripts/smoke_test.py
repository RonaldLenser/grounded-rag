"""End-to-end smoke test against a RUNNING Grounded server.

Unlike the pytest suite (which fakes the embedding provider), this uses the
real stack: real server, real database, real OpenAI embeddings. Use it to
check a local Docker run or a deployed instance.

Usage:
    uv run python scripts/smoke_test.py                       # http://localhost:8000
    uv run python scripts/smoke_test.py https://your-app.run.app

Reads ADMIN_API_KEY from the environment or from .env. Exit code 0 = all passed.
Cost: one tiny embedding request (fractions of a cent).
"""

import os
import sys
import time
from pathlib import Path

import httpx2

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample.pdf"
TIMEOUT_SECONDS = 60.0


def load_admin_key() -> str:
    if key := os.environ.get("ADMIN_API_KEY"):
        return key
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ADMIN_API_KEY" and value.strip():
                return value.strip()
    sys.exit("FAIL  ADMIN_API_KEY not found in environment or .env")


class SmokeTest:
    def __init__(self, base_url: str, admin_key: str) -> None:
        self.client = httpx2.Client(
            base_url=base_url, headers={"X-API-Key": admin_key}, timeout=TIMEOUT_SECONDS
        )
        self.failures = 0

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        print(f"{status}  {label}" + (f"  ({detail})" if detail else ""))
        if not passed:
            self.failures += 1
        return passed

    def run(self) -> int:
        print(f"Smoke testing {self.client.base_url}\n")

        try:
            ready = self.client.get("/ready")
        except httpx2.HTTPError as exc:
            self.check("server is reachable", False, f"{type(exc).__name__}: is it running?")
            return 1
        if not self.check(
            "server ready + database connected", ready.status_code == 200, _fail(ready)
        ):
            return 1

        name = f"smoke-test-{int(time.time())}"
        created = self.client.post("/v1/collections", json={"name": name})
        if not self.check(
            "create collection (admin key accepted)", created.status_code == 201, _fail(created)
        ):
            return 1
        collection_id = created.json()["id"]

        started = time.perf_counter()
        uploaded = self.client.post(
            f"/v1/collections/{collection_id}/documents",
            files={"file": ("sample.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
        )
        seconds = time.perf_counter() - started
        if not self.check(
            "upload PDF: parse -> chunk -> REAL OpenAI embeddings -> store",
            uploaded.status_code == 201,
            _fail(uploaded),
        ):
            return 1
        document = uploaded.json()
        self.check(
            "document is ready with chunks",
            document["status"] == "ready" and document["chunk_count"] > 0,
            f"{document['page_count']} pages, {document['chunk_count']} chunks, {seconds:.1f}s",
        )

        fetched = self.client.get(f"/v1/documents/{document['id']}")
        self.check("fetch document by id", fetched.status_code == 200)

        listed = self.client.get(f"/v1/collections/{collection_id}/documents")
        self.check(
            "document appears in collection list",
            listed.status_code == 200 and [d["id"] for d in listed.json()] == [document["id"]],
        )

        relevant = self._top_score(
            collection_id, "How is AI risk measured with quantitative tools?"
        )
        unrelated = self._top_score(collection_id, "best chocolate chip cookie recipe")
        self.check(
            "search ranks a relevant question above an unrelated one (real embeddings)",
            relevant is not None and unrelated is not None and relevant > unrelated,
            f"relevant={relevant}, unrelated={unrelated}",
        )

        duplicate = self.client.post(
            f"/v1/collections/{collection_id}/documents",
            files={"file": ("renamed.pdf", SAMPLE_PDF.read_bytes(), "application/pdf")},
        )
        self.check(
            "same file again is rejected as duplicate",
            duplicate.status_code == 409
            and duplicate.json()["error"]["code"] == "duplicate_document",
        )

        unauthorized = self.client.post(
            "/v1/collections", json={"name": "nope"}, headers={"X-API-Key": "wrong-key"}
        )
        self.check("wrong admin key is rejected", unauthorized.status_code == 401)

        deleted = self.client.delete(f"/v1/documents/{document['id']}")
        gone = self.client.get(f"/v1/documents/{document['id']}")
        self.check("delete document", deleted.status_code == 204 and gone.status_code == 404)

        print()
        if self.failures:
            print(f"{self.failures} CHECK(S) FAILED")
            return 1
        print("ALL CHECKS PASSED")
        return 0

    def _top_score(self, collection_id: str, query: str) -> float | None:
        response = self.client.post(
            f"/v1/collections/{collection_id}/search", json={"query": query, "top_k": 1}
        )
        results = response.json().get("results") if response.status_code == 200 else None
        return round(results[0]["score"], 3) if results else None


def _fail(response: httpx2.Response) -> str:
    """Response details, shown only when a check fails."""
    if response.is_success:
        return ""
    return f"HTTP {response.status_code}: {response.text[:300]}"


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    sys.exit(SmokeTest(url.rstrip("/"), load_admin_key()).run())
