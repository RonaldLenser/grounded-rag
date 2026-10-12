"""Load the public demo corpus into a running Grounded server.

Corpus: NIST AI Risk Management Framework documents (US government works,
public domain). Downloaded from their permanent DOI links and cached in
evals/corpus/ (gitignored), so the M7 evals can reuse the same files.

Safe to re-run: an existing collection is reused and duplicate uploads
(same content hash) are skipped.

--replace deletes the demo documents first and uploads them again. Use it
after changing parsing or chunking: identical files would otherwise be skipped
as duplicates and keep their old chunks. The demo has no documents for the
minute or so this takes.

Usage:
    uv run python scripts/seed_demo.py                         # http://localhost:8000
    uv run python scripts/seed_demo.py https://your-app.run.app
    uv run python scripts/seed_demo.py https://your-app.run.app --replace

Reads ADMIN_API_KEY from the environment or .env.
Cost: embedding ~100 pages, well under one cent.
"""

import argparse
import sys
from pathlib import Path

import httpx2
from smoke_test import load_admin_key

COLLECTION = "nist-ai-rmf"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "evals" / "corpus"
DOCUMENTS = {
    "nist-ai-rmf-1.0.pdf": "https://doi.org/10.6028/NIST.AI.100-1",
    "nist-ai-600-1-genai-profile.pdf": "https://doi.org/10.6028/NIST.AI.600-1",
}


def download(filename: str, url: str) -> bytes:
    path = CORPUS_DIR / filename
    if path.exists():
        return path.read_bytes()
    print(f"  downloading {filename} ...")
    response = httpx2.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF-"):
        sys.exit(f"FAIL  {url} did not return a PDF")
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def find_or_create_collection(client: httpx2.Client) -> str:
    for collection in client.get("/v1/collections").raise_for_status().json():
        if collection["name"] == COLLECTION:
            print(f"Using existing collection '{COLLECTION}'")
            return str(collection["id"])
    created = client.post("/v1/collections", json={"name": COLLECTION})
    if created.status_code != 201:
        sys.exit(f"FAIL  create collection: HTTP {created.status_code} {created.text[:300]}")
    print(f"Created collection '{COLLECTION}'")
    return str(created.json()["id"])


def delete_demo_documents(client: httpx2.Client, collection_id: str) -> None:
    documents = client.get(f"/v1/collections/{collection_id}/documents").raise_for_status().json()
    for doc in documents:
        if doc["filename"] in DOCUMENTS:
            client.delete(f"/v1/documents/{doc['id']}").raise_for_status()
            print(f"  deleted {doc['filename']} (will re-upload)")


def main(base_url: str, *, replace: bool = False) -> int:
    client = httpx2.Client(base_url=base_url, headers={"X-API-Key": load_admin_key()}, timeout=300)
    print(f"Seeding {base_url}\n")
    collection_id = find_or_create_collection(client)
    if replace:
        delete_demo_documents(client, collection_id)

    for filename, url in DOCUMENTS.items():
        data = download(filename, url)
        response = client.post(
            f"/v1/collections/{collection_id}/documents",
            files={"file": (filename, data, "application/pdf")},
        )
        if response.status_code == 201:
            doc = response.json()
            print(f"  added {filename}: {doc['page_count']} pages, {doc['chunk_count']} chunks")
        elif response.status_code == 409:
            print(f"  skipped {filename}: already uploaded")
        else:
            print(f"FAIL  {filename}: HTTP {response.status_code} {response.text[:300]}")
            return 1

    print(f"\nDone. Collection id: {collection_id}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load the NIST demo corpus.")
    parser.add_argument("base_url", nargs="?", default="http://localhost:8000")
    parser.add_argument(
        "--replace", action="store_true", help="delete and re-upload the demo documents"
    )
    args = parser.parse_args()
    sys.exit(main(args.base_url.rstrip("/"), replace=args.replace))
