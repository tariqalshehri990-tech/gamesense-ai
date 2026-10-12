"""Shared match state ("the fact layer") and the tool surface exposed over MCP.

The same `MatchFacts` object backs three things: the agents (in-process tool calls), the MCP
server (`nafas.mcp_server`), and the HTTP API. Numbers always originate here, never in a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .simulator import MATCH_SECONDS, TEAMS, Match, Player, Shot
from .stats import Book


@dataclass
class MatchFacts:
    seed: int
    match: Match
    players: dict[str, Player]
    book: Book
    ledger_timeline: list[dict[str, Any]] = field(default_factory=list)
    baseline_rows: dict[str, dict] = field(default_factory=dict)
    moments: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    # ---- helpers --------------------------------------------------------------------
    def _event(self, eid: str) -> dict | None:
        return next((e for e in self.match.events if e["id"] == eid), None)

    def _shot(self, sid: str) -> Shot | None:
        return next((s for s in self.match.shots if s.id == sid), None)

    def name(self, pid: str, lang: str = "en") -> str:
        p = self.players[pid]
        return p.name_ar if lang == "ar" else p.name

    # ---- tools (each returns plain JSON-able data) -----------------------------------
    def get_match_state(self, t: float = MATCH_SECONDS) -> dict:
        """Score, shots and cumulative xG per team up to match second `t`."""
        out = {tm: {"goals": 0, "shots": 0, "on_target": 0, "xg": 0.0} for tm in TEAMS}
        for s in self.match.shots:
            if s.t <= t:
                o = out[s.team]
                o["shots"] += 1
                o["on_target"] += int(s.on_target)
                o["goals"] += int(s.goal)
                o["xg"] = round(o["xg"] + s.xg, 3)
        return {"t": t, "teams": out}

    def get_shot(self, shot_id: str) -> dict:
        """Full record for one shot, with shooter and keeper names."""
        s = self._shot(shot_id)
        if s is None:
            return {"error": f"unknown shot {shot_id}"}
        r = s.record()
        r["shooter_name"] = self.players[s.shooter].name
        r["keeper_name"] = self.players[s.keeper].name
        return r

    def get_player_value(self, player_id: str, t: float | None = None,
                         before_shot: str | None = None) -> dict:
        """Ledger row (skill/luck split, pulse band, evidence gate) as of match second `t`.

        With `before_shot` it returns the row as it stood just before that shot was counted."""
        if player_id not in self.players:
            return {"error": f"unknown player {player_id}"}
        if before_shot is not None:
            for snap in self.ledger_timeline:
                if snap["shot_id"] == before_shot and player_id in snap["before"]:
                    return snap["before"][player_id]
            return {"error": "no snapshot"}
        row = self.baseline_rows.get(player_id)
        for snap in self.ledger_timeline:
            if t is not None and snap["t"] > t:
                break
            if player_id in snap["rows"]:
                row = snap["rows"][player_id]
        return row or {"error": "no data"}

    def get_evidence(self, event_ids: list[str]) -> list[dict]:
        """The raw events behind a claim."""
        return [e for e in (self._event(i) for i in event_ids) if e]

    def list_moments(self, min_importance: float = 0.0) -> list[dict]:
        return [m for m in self.moments if m["importance"] >= min_importance]

    def whatif(self, shot_id: str, goal: bool) -> dict:
        """Flip one shot's outcome. Labelled simulation, not what would have happened."""
        try:
            return self.book.whatif(shot_id, goal)
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}


TOOL_SPECS = {
    "get_match_state": "Score, shots and xG per team up to a match second.",
    "get_shot": "Full record for one shot id (features, xG, post-shot xG, outcome).",
    "get_player_value": "Skill-vs-luck split, contract pulse band and evidence gate for a player.",
    "get_evidence": "Raw events behind a claim, by event id.",
    "list_moments": "Detected moments above an importance threshold.",
    "whatif": "Flip a shot outcome and see the value change (simulation scenario).",
}


def call_tool(facts: MatchFacts, name: str, **kwargs: Any) -> Any:
    """In-process tool call with a trace entry, mirroring what an MCP client would do."""
    if name not in TOOL_SPECS:
        raise KeyError(name)
    result = getattr(facts, name)(**kwargs)
    facts.tool_log.append({"tool": name, "args": kwargs})
    return result
