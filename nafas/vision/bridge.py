"""Video-derived events -> the Nafas facts layer (xG, ledger, moments, commentary)."""
from __future__ import annotations

from ..agents import HISTORY_MATCHES, facts_for_match
from ..simulator import Match, Shot, build_league, simulate_season
from ..stats import Book, calibrate_goal_value
from .. import xg as XG
from .events import Event
from .pitch import GOAL_W, PITCH_W

SIDE_EPS = 1.5


def match_from_events(seed: int, events: list[Event], t0: float, players) -> Match:
    """Build a Match whose shots and events come from the tracked video. `t0` is the clip's match-clock offset (s)."""
    match = Match(index=HISTORY_MATCHES, seed=seed, players=players)
    opp = {"KBY": "RMR", "RMR": "KBY"}
    keepers = {t: next(p.id for p in players.values() if p.team == t and p.role == "GK") for t in opp}
    passes = [e for e in events if e.kind == "pass"]
    seq = 0

    def add(kind: str, t: float, **kw) -> None:
        nonlocal seq
        seq += 1
        match.events.append({"id": f"V{seq:04d}", "t": round(t0 + t, 1), "type": kind, **kw})

    n = 0
    for e in events:
        if e.kind == "pass":
            add("pass", e.t, team=e.team, player=e.player, to=e.detail["to"], length_m=e.detail["length_m"])
        elif e.kind == "shot":
            n += 1
            d = e.detail
            assister = next((p.player for p in reversed(passes) if p.detail["to"] == e.player and 0 < e.t - p.t < 8), None)
            feats = XG.ShotFeatures(distance=d["distance"], angle=d["angle"], pressure=d["pressure"],
                                    assist="pass" if assister else "none")
            on_target = d["outcome"] in ("goal", "save")
            placement = psx = None
            if on_target:
                placement = min(1.0, abs(d["aim_y"] - PITCH_W / 2) / (GOAL_W / 2))
                psx = XG.psxg(feats, placement)
            goal = d["outcome"] == "goal"
            lat = d["y"] - PITCH_W / 2
            shot = Shot(id=f"S{HISTORY_MATCHES:02d}{n:03d}", match=HISTORY_MATCHES, t=t0 + e.t, team=e.team,
                        shooter=e.player, keeper=keepers[opp[e.team]], assister=assister, feats=feats,
                        xg=XG.xg(feats), on_target=on_target, placement=placement, psxg=psx, goal=goal,
                        saved=d["outcome"] == "save", side="left" if lat < -SIDE_EPS else "right" if lat > SIDE_EPS else "centre")
            match.shots.append(shot)
            add("shot", e.t, team=e.team, player=e.player, shot_id=shot.id, xg=round(shot.xg, 3), on_target=on_target,
                distance=d["distance"], side=shot.side, head=False, penalty=False)
            if on_target:
                add("goal" if goal else "save", e.t + 1.0, team=e.team if goal else opp[e.team],
                    player=e.player if goal else shot.keeper, shot_id=shot.id, psxg=round(psx, 3), placement=round(placement, 2))
                if goal:
                    match.score[e.team] += 1
            else:
                add("off_target", e.t + 1.0, team=e.team, player=e.player, shot_id=shot.id)
    match.events.sort(key=lambda x: (x["t"], x["id"]))
    return match


def facts_from_video(seed: int, events: list[Event], t0: float):
    players = build_league(seed)
    book = Book.from_history(players, simulate_season(seed, HISTORY_MATCHES))
    calibrate_goal_value(book)
    return facts_for_match(seed, players, book, match_from_events(seed, events, t0, players))
