"""HTTP API and live stream for the Nafas demo.

  GET  /                      the demo UI
  GET  /api/health            liveness
  GET  /api/session           full deterministic session as JSON (all profiles)
  GET  /api/stream            Server-Sent Events, paced in match time (speed=60 -> 90 min in 90 s)
  GET  /api/ledger            final investor ledger
  POST /api/whatif            flip one shot's outcome (simulation scenario)
"""
from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents import build_facts, build_session, make_backend
from .attention import PROFILES
from .facts import MatchFacts

DEFAULT_SEED = int(os.environ.get("NAFAS_SEED", "35"))
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Nafas", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_backend = make_backend()


@lru_cache(maxsize=16)
def _facts(seed: int) -> MatchFacts:
    return build_facts(seed)


_sessions: dict[tuple[int, str], dict[str, Any]] = {}


async def _session(seed: int, lang: str | None) -> dict[str, Any]:
    key = (seed, lang or "")
    if key not in _sessions:
        if len(_sessions) > 24:
            _sessions.clear()
        _sessions[key] = await build_session(seed, backend=_backend, lang=lang, facts=_facts(seed))
    return _sessions[key]


def _check(seed: int, lang: str | None, profile: str | None = None) -> None:
    if not 0 <= seed <= 10_000:
        raise HTTPException(400, "seed must be between 0 and 10000")
    if lang not in (None, "", "ar", "en"):
        raise HTTPException(400, "lang must be ar or en")
    if profile is not None and profile not in PROFILES:
        raise HTTPException(400, f"profile must be one of {list(PROFILES)}")


def _items(session: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    prof = session["profiles"][profile]
    names = {p["id"]: p for p in session["meta"]["players"]}
    out: list[dict[str, Any]] = []
    for e in session["events"]:
        if e["type"] in ("shot", "goal", "save", "off_target"):
            p = names.get(e.get("player"), {})
            out.append({"type": "event", "t": e["t"], "kind": e["type"], "team": e.get("team"),
                        "player": p.get("name"), "player_ar": p.get("name_ar"), "xg": e.get("xg"),
                        "shot_id": e.get("shot_id"), "id": e["id"]})
    for u in prof["utterances"]:
        out.append({"type": "utterance", "t": u["start"], **u})
    for d in prof["dropped"]:
        out.append({"type": "dropped", **d})
    for snap in session["ledger_timeline"]:
        out.append({"type": "ledger", "t": snap["t"], "shot_id": snap["shot_id"], "rows": snap["rows"],
                    "before": snap["before"]})
    for tr in session["trace"]:
        out.append({"type": "trace", **tr})
    out.sort(key=lambda x: (x["t"], {"event": 0, "ledger": 1, "utterance": 2}.get(x["type"], 3)))
    return out


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "backend": _backend.name, "default_seed": DEFAULT_SEED}


@app.get("/api/session")
async def session(seed: int = DEFAULT_SEED, lang: str | None = Query(None)) -> dict[str, Any]:
    _check(seed, lang)
    return await _session(seed, lang)


@app.get("/api/ledger")
async def ledger(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    _check(seed, None)
    return (await _session(seed, None))["ledger_final"]


@app.get("/api/stream")
async def stream(seed: int = DEFAULT_SEED, profile: str = "casual", lang: str | None = Query(None),
                 speed: float = Query(60.0, gt=0, le=1_000_000)) -> StreamingResponse:
    _check(seed, lang, profile)
    sess = await _session(seed, lang)

    async def gen():
        meta = sess["meta"]
        yield _sse({"type": "meta", "t": 0, "meta": meta, "profile": profile,
                    "baseline": sess["ledger_baseline"], "label": sess["profiles"][profile]["label"],
                    "lang": sess["profiles"][profile]["lang"]})
        last = 0.0
        for it in _items(sess, profile):
            dt = (it["t"] - last) / speed
            if dt > 0.001:
                await asyncio.sleep(min(dt, 4.0))
            last = max(last, it["t"])
            yield _sse(it)
        yield _sse({"type": "end", "t": meta["duration"], "metrics": sess["profiles"][profile]["metrics"],
                    "ledger_final": sess["ledger_final"], "conductor": sess["conductor"]})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class WhatIf(BaseModel):
    seed: int = DEFAULT_SEED
    shot_id: str
    goal: bool


@app.post("/api/whatif")
async def whatif(body: WhatIf) -> dict[str, Any]:
    _check(body.seed, None)
    res = _facts(body.seed).whatif(body.shot_id, body.goal)
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res
