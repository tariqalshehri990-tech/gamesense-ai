import asyncio
from fastapi.testclient import TestClient
from nafas.server import app

c = TestClient(app)


def test_health_and_validation():
    assert c.get("/api/health").json()["ok"] is True
    assert c.get("/api/session?seed=99999").status_code == 400
    assert c.get("/api/stream?profile=nope").status_code == 400


def test_stream_is_ordered_and_ends():
    r = c.get("/api/stream?seed=35&profile=casual&speed=1000000")
    import json
    items = [json.loads(l[6:]) for l in r.text.split("\n\n") if l.startswith("data: ")]
    assert items[0]["type"] == "meta" and items[-1]["type"] == "end"
    ts = [i["t"] for i in items]
    assert ts == sorted(ts)


def test_whatif_unknown_shot():
    assert c.post("/api/whatif", json={"seed": 35, "shot_id": "nope", "goal": True}).status_code == 400


def test_mcp_tools():
    from nafas.mcp_server import create_server
    srv = create_server()
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert {"get_match_state", "get_shot", "get_player_value", "get_evidence", "list_moments", "whatif"} <= names
