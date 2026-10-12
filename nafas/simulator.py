"""Seeded synthetic match generator: fictional clubs, fictional players, fictional contracts.

No real player, club, contract or match data is used anywhere. Every number is generated.
Each player has a hidden true finishing / keeping skill that the analytics layer must *estimate*
from events, which lets us check that the model recovers it (see tests/test_stats.py).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

from . import xg as XG

MATCH_SECONDS = 90 * 60
SHOTS_PER_TEAM_MEAN = 10.5
FINISH_TAU = 0.20   # SD of true finishing skill on the logit scale (assumption, see README)
KEEPER_TAU = 0.15


@dataclass
class Player:
    id: str
    name: str
    name_ar: str
    team: str
    role: str                 # GK / DEF / MID / FWD
    finishing: float = 0.0    # hidden true skill (logit offset)
    keeping: float = 0.0      # hidden true skill (logit offset)
    wage_k_week: float = 0.0  # fictional weekly wage, thousands
    contract_years: int = 0
    shot_weight: float = 1.0  # share of the team's shots taken

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("finishing")
        d.pop("keeping")
        d.pop("shot_weight")
        return d


TEAMS = {
    "KBY": {"name": "Kestrel Bay", "name_ar": "كيستريل باي"},
    "RMR": {"name": "Redmoor Rovers", "name_ar": "ريدمور روفرز"},
}

# (id, name, name_ar, role, wage_k_week, contract_years, shot_weight)
_ROSTER = {
    "KBY": [
        ("kby_gk", "Tomas Brandt", "توماس براندت", "GK", 95, 3, 0.0),
        ("kby_d1", "Ilan Costa", "إيلان كوستا", "DEF", 70, 2, 0.35),
        ("kby_d2", "Marek Doyle", "مارك دويل", "DEF", 85, 4, 0.40),
        ("kby_m1", "Soren Alvik", "سورين ألفيك", "MID", 140, 4, 1.4),
        ("kby_m2", "Caio Renner", "كايو رينر", "MID", 110, 3, 1.1),
        ("kby_m3", "Yusuf Halden", "يوسف هالدن", "MID", 60, 1, 0.9),
        ("kby_f1", "Dario Valk", "داريو فالك", "FWD", 260, 5, 3.0),
        ("kby_f2", "Nils Okoro", "نيلز أوكورو", "FWD", 120, 2, 2.2),
    ],
    "RMR": [
        ("rmr_gk", "Anton Weiss", "أنتون فايس", "GK", 150, 4, 0.0),
        ("rmr_d1", "Felix Moran", "فيليكس موران", "DEF", 75, 3, 0.35),
        ("rmr_d2", "Idris Maro", "إدريس مارو", "DEF", 65, 2, 0.40),
        ("rmr_m1", "Lukas Pryce", "لوكاس برايس", "MID", 130, 3, 1.4),
        ("rmr_m2", "Theo Navarro", "ثيو نافارو", "MID", 90, 2, 1.1),
        ("rmr_m3", "Omar Lindqvist", "عمر ليندكفيست", "MID", 55, 1, 0.9),
        ("rmr_f1", "Rafael Stroud", "رافاييل ستراود", "FWD", 310, 5, 3.0),
        ("rmr_f2", "Jonas Ekwall", "يوناس إكفال", "FWD", 80, 1, 2.2),
    ],
}


def build_league(seed: int) -> dict[str, Player]:
    """Roster with hidden skills drawn from the seed (stable across all matches of a seed)."""
    rng = random.Random(f"league-{seed}")
    players: dict[str, Player] = {}
    for team, rows in _ROSTER.items():
        for pid, name, name_ar, role, wage, yrs, w in rows:
            players[pid] = Player(
                id=pid, name=name, name_ar=name_ar, team=team, role=role,
                finishing=rng.gauss(0, FINISH_TAU) if role != "GK" else 0.0,
                keeping=rng.gauss(0, KEEPER_TAU) if role == "GK" else 0.0,
                wage_k_week=wage, contract_years=yrs, shot_weight=w,
            )
    return players


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _draw_features(rng: random.Random) -> XG.ShotFeatures:
    if rng.random() < 0.02:
        return XG.ShotFeatures(distance=11.0, angle=0.62, penalty=True, assist="none")
    # distance: log-normal around ~17 m, clipped
    dist = min(max(rng.lognormvariate(math.log(17.0), 0.38), 5.0), 38.0)
    lateral = rng.gauss(0, 7.5)
    # angle subtended by the 7.32 m goal from (dist, lateral)
    a1 = math.atan2(lateral + 3.66, dist)
    a2 = math.atan2(lateral - 3.66, dist)
    angle = abs(a1 - a2)
    head = rng.random() < 0.18
    pressure = rng.choices([0, 1, 2], weights=[0.40, 0.38, 0.22])[0]
    assist = rng.choices(["none", "pass", "cross", "through", "set_piece"],
                         weights=[0.14, 0.42, 0.16, 0.12, 0.16])[0]
    if head and assist == "none":
        assist = "cross"
    return XG.ShotFeatures(distance=dist, angle=angle, head=head, pressure=pressure, assist=assist)


def _side(lateral_hint: float) -> str:
    return "left" if lateral_hint < -1.5 else ("right" if lateral_hint > 1.5 else "centre")


@dataclass
class Shot:
    id: str
    match: int
    t: float
    team: str
    shooter: str
    keeper: str
    assister: str | None
    feats: XG.ShotFeatures
    xg: float
    on_target: bool
    placement: float | None
    psxg: float | None
    goal: bool
    saved: bool
    side: str

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id, "match": self.match, "t": self.t, "team": self.team,
            "shooter": self.shooter, "keeper": self.keeper, "assister": self.assister,
            "xg": round(self.xg, 4), "on_target": self.on_target,
            "placement": None if self.placement is None else round(self.placement, 3),
            "psxg": None if self.psxg is None else round(self.psxg, 4),
            "goal": self.goal, "saved": self.saved, "side": self.side,
            "distance": round(self.feats.distance, 1), "head": self.feats.head,
            "pressure": self.feats.pressure, "assist_type": self.feats.assist,
            "penalty": self.feats.penalty,
        }


@dataclass
class Match:
    index: int
    seed: int
    players: dict[str, Player]
    events: list[dict[str, Any]] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    score: dict[str, int] = field(default_factory=lambda: {"KBY": 0, "RMR": 0})
    chaos_windows: list[tuple[float, float]] = field(default_factory=list)


def _pick(rng: random.Random, players: list[Player], weight=lambda p: p.shot_weight) -> Player:
    ws = [max(weight(p), 0.0001) for p in players]
    return rng.choices(players, weights=ws)[0]


def simulate_match(seed: int, index: int = 0, players: dict[str, Player] | None = None,
                   with_filler: bool = True) -> Match:
    """Simulate one match. `index` makes the match unique inside a season."""
    players = players or build_league(seed)
    rng = random.Random(f"match-{seed}-{index}")
    match = Match(index=index, seed=seed, players=players)
    by_team = {t: [p for p in players.values() if p.team == t] for t in TEAMS}
    keepers = {t: next(p for p in by_team[t] if p.role == "GK") for t in TEAMS}

    # --- shot times, with 1-2 "chaos" windows where shots cluster ---------------------------
    shot_times: list[tuple[float, str]] = []
    for team in TEAMS:
        n = _poisson(rng, SHOTS_PER_TEAM_MEAN)
        shot_times += [(rng.uniform(30, MATCH_SECONDS - 30), team) for _ in range(n)]
    for _ in range(rng.choice([1, 2])):
        start = rng.uniform(600, MATCH_SECONDS - 400)
        match.chaos_windows.append((start, start + 45))
        team = rng.choice(list(TEAMS))
        for _ in range(rng.choice([3, 4])):
            shot_times.append((rng.uniform(start, start + 45), team))
    shot_times.sort()

    seq = 0

    def add(kind: str, t: float, **kw: Any) -> dict[str, Any]:
        nonlocal seq
        seq += 1
        ev = {"id": f"M{index:02d}E{seq:04d}", "t": round(t, 1), "type": kind, **kw}
        match.events.append(ev)
        return ev

    last_by_team: dict[str, tuple[Player, float]] = {}
    for n_shot, (t, team) in enumerate(shot_times, 1):
        opp = "RMR" if team == "KBY" else "KBY"
        shooter = _pick(rng, [p for p in by_team[team] if p.role != "GK"])
        feats = _draw_features(rng)
        if feats.penalty:
            shooter = max((p for p in by_team[team] if p.role != "GK"), key=lambda p: p.shot_weight)
        assister = None
        if feats.assist in ("pass", "cross", "through"):
            cands = [p for p in by_team[team] if p.role != "GK" and p.id != shooter.id]
            assister = _pick(rng, cands, weight=lambda p: 1.0 + (p.role == "MID")).id
        s_q = XG.shot_quality(feats)
        pot = XG.p_on_target(feats)
        on_target = rng.random() < pot
        keeper = keepers[opp]
        placement = psx = None
        goal = saved = False
        if on_target:
            placement = rng.betavariate(2, 2)
            psx = XG.psxg(feats, placement)
            p_goal = XG.sigmoid(XG.logit(psx) + shooter.finishing - keeper.keeping)
            goal = rng.random() < p_goal
            saved = not goal
        lateral_hint = rng.gauss(0, 7.5)
        shot = Shot(
            id=f"S{index:02d}{n_shot:03d}", match=index, t=t, team=team, shooter=shooter.id,
            keeper=keeper.id, assister=assister, feats=feats, xg=XG.xg(feats),
            on_target=on_target, placement=placement, psxg=psx, goal=goal, saved=saved,
            side=_side(lateral_hint),
        )
        match.shots.append(shot)
        if assister:
            add("pass", t - 4, team=team, player=assister, to=shooter.id, key_pass=True,
                pass_type=feats.assist)
        add("shot", t, team=team, player=shooter.id, shot_id=shot.id, xg=round(shot.xg, 3),
            on_target=on_target, distance=round(feats.distance, 1), side=shot.side,
            head=feats.head, penalty=feats.penalty)
        if on_target:
            add("save" if saved else "goal", t + 1.2, team=team if goal else opp,
                player=shooter.id if goal else keeper.id, shot_id=shot.id,
                psxg=round(psx, 3), placement=round(placement, 2))
            if goal:
                match.score[team] += 1
        else:
            add("off_target", t + 1.0, team=team, player=shooter.id, shot_id=shot.id)

    if with_filler:
        _filler(rng, match, by_team)
    match.events.sort(key=lambda e: (e["t"], e["id"]))
    return match


def _filler(rng: random.Random, match: Match, by_team: dict[str, list[Player]]) -> None:
    """Context events: pressure spells, tackles, possession changes. Used for narrative colour
    and for pattern detection (HIGH_PRESS). They carry no outcome information."""
    seq = len(match.events)

    def add(kind: str, t: float, **kw: Any) -> None:
        nonlocal seq
        seq += 1
        match.events.append({"id": f"M{match.index:02d}E{seq:04d}", "t": round(t, 1),
                             "type": kind, **kw})

    # high-press spells: 3-5 minutes in which a team piles up pressure events
    for _ in range(rng.choice([2, 3, 4])):
        team = rng.choice(list(TEAMS))
        start = rng.uniform(300, MATCH_SECONDS - 400)
        for _ in range(rng.randint(7, 11)):
            p = _pick(rng, [q for q in by_team[team] if q.role in ("MID", "FWD")], weight=lambda q: 1)
            add("pressure", start + rng.uniform(0, 200), team=team, player=p.id, zone="final_third")
    # background tackles and possession changes
    for _ in range(46):
        team = rng.choice(list(TEAMS))
        p = _pick(rng, [q for q in by_team[team] if q.role != "GK"], weight=lambda q: 1)
        add("tackle", rng.uniform(20, MATCH_SECONDS - 20), team=team, player=p.id,
            won=rng.random() < 0.6)
    t = 0.0
    team = rng.choice(list(TEAMS))
    while t < MATCH_SECONDS - 60:
        t += rng.uniform(40, 110)
        team = "RMR" if team == "KBY" else "KBY"
        add("possession", t, team=team, zone=rng.choice(["defensive", "middle", "attacking"]))


def simulate_season(seed: int, n_matches: int = 38) -> list[Match]:
    """A synthetic history: the evidence base the investor mode reasons over."""
    players = build_league(seed)
    return [simulate_match(seed, i, players, with_filler=False) for i in range(n_matches)]
