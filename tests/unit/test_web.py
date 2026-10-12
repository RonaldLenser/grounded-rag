from fastapi.testclient import TestClient

from grounded.api.routes.web import WEB_DIR


def test_index_is_served_with_security_headers(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Grounded</title>" in response.text
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_static_assets_are_served(client: TestClient) -> None:
    js = client.get("/static/app.js")
    css = client.get("/static/styles.css")

    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]


def test_index_is_hidden_from_the_api_schema(client: TestClient) -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_ui_never_uses_innerhtml() -> None:
    """XSS guard: answers, quotes and filenames are untrusted. They must be rendered
    with textContent / DOM nodes. This fails if someone adds innerHTML later."""
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert dangerous not in source


def test_index_has_no_inline_scripts_or_styles() -> None:
    """The CSP blocks inline code; inline <script>/<style> would silently break the page."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert "<script>" not in html
    assert "<style" not in html
    assert " onclick=" not in html
