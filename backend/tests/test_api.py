import pytest


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/api/health/ping")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


