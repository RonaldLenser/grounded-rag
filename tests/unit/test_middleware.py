from fastapi.testclient import TestClient

HEADER = "X-Request-ID"


def test_request_id_is_generated_when_missing(client: TestClient) -> None:
    response = client.get("/health")

    request_id = response.headers[HEADER]
    assert len(request_id) == 32  # uuid4 hex


def test_each_request_gets_a_different_id(client: TestClient) -> None:
    first = client.get("/health").headers[HEADER]
    second = client.get("/health").headers[HEADER]

    assert first != second


def test_safe_incoming_request_id_is_reused(client: TestClient) -> None:
    response = client.get("/health", headers={HEADER: "trace-abc_123"})

    assert response.headers[HEADER] == "trace-abc_123"


def test_unsafe_incoming_request_id_is_replaced(client: TestClient) -> None:
    # Spaces and symbols could be used to forge or corrupt log lines.
    response = client.get("/health", headers={HEADER: "evil id; drop logs"})

    assert response.headers[HEADER] != "evil id; drop logs"
    assert len(response.headers[HEADER]) == 32


def test_overlong_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={HEADER: "a" * 65})

    assert len(response.headers[HEADER]) == 32


def test_request_id_is_present_on_404s(client: TestClient) -> None:
    response = client.get("/missing")

    assert response.status_code == 404
    assert HEADER in response.headers
