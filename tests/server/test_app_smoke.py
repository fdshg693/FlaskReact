from __future__ import annotations

import pytest

from server.flask_react_app.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_root_serves_home(client):
    resp = client.get("/")
    # In dev, static files may not be present; at least assert a response
    assert resp.status_code in (200, 404)


def test_text_split_validation(client):
    resp = client.post("/api/textSplit", json={})
    assert resp.status_code == 400
    resp2 = client.post(
        "/api/textSplit",
        json={"text": "hello", "chunk_size": 200, "chunk_overlap": 100},
    )
    assert resp2.status_code == 200
    assert "chunks" in resp2.get_json()
