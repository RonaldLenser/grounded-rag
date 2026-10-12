"""Serves the single-page web UI (src/grounded/web/).

The page is static HTML/CSS/JS that calls the same public API as any client.
It sends a strict Content-Security-Policy: the browser may only run scripts
and styles from this server, so even if untrusted text slipped into the page,
injected inline scripts would not execute (defense in depth on top of the
UI's textContent-only rendering).
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

router = APIRouter(include_in_schema=False)


@router.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers=SECURITY_HEADERS)
