"""Investor-mode analytics: skill-vs-luck, contract pulse, counterfactuals, evidence gate.

Design rules (see README "What we refuse to claim"):
  * every estimate carries an interval; the point value is never shown alone
  * below an evidence threshold the system says "no verdict" instead of a number
  * the goals-to-money conversion is a labelled synthetic assumption, calibrated so the median
    synthetic starter is break-even. Only relative comparisons carry meaning.
  * descriptive, not predictive; never buy/sell/overpaid language
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from . import xg as XG
from .simulator import FINISH_TAU, KEEPER_TAU, Match, Player, Shot

# --- design assumptions (documented, adjustable) ------------------------------------------
GATE_DIRECTIONAL = 100   # shots (or shots faced) before a skill estimate is even directional
GATE_USABLE = 250        # shots before it is treated as usable
ASSIST_WEIGHT = 0.6      # share of an assisted chance's xG credited to the passer
DEF_ACTION_VALUE = 0.015 # goal-value of a won tackle
REPLACEMENT_PER_MATCH = {"FWD": 0.22, "MID": 0.10, "DEF": 0.05, "GK": -0.03}
GOAL_VALUE_K = 1500.0    # synthetic $k per goal-equivalent above replacement (see calibrate)
WEEKS_PER_YEAR = 52

_THETAS = [(-1.0 + 0.01 * i) for i in range(201)]


def gate_for(n: int) -> str:
    if n < GATE_DIRECTIONAL:
        return "insufficient"
    return "directional" if n < GATE_USABLE else "usable"


@dataclass
class Interval:
    mean: float
    lo: float
    hi: float

    def as_dict(self, nd: int = 3) -> dict[str, float]:
        return {"mean": round(self.mean, nd), "lo": round(self.lo, nd), "hi": round(self.hi, nd)}


def _grid_posterior(loglik: list[float], tau: float) -> list[float]:
    lp = [ll - 0.5 * (t / tau) ** 2 for ll, t in zip(loglik, _THETAS)]
    m = max(lp)
    w = [math.exp(x - m) for x in lp]
    s = sum(w)
    return [x / s for x in w]


def _quantile(post: list[float], q: float) -> float:
    c = 0.0
    for t, w in zip(_THETAS, post):
        c += w
        if c >= q:
            return t
    return _THETAS[-1]


def _ll_contrib(base: float, y: int, sign: float) -> list[float]:
    out = []
    for th in _THETAS:
        p = min(max(XG.sigmoid(base + sign * th), 1e-9), 1 - 1e-9)
        out.append(math.log(p) if y else math.log(1 - p))
    return out


def _ll_vector(base_logits: list[float], outcomes: list[int], sign: float) -> list[float]:
    ll = [0.0] * len(_THETAS)
    for b, y in zip(base_logits, outcomes):
        c = _ll_contrib(b, y, sign)
        ll = [a + x for a, x in zip(ll, c)]
    return ll


def _skill_posterior(base_logits, outcomes, tau, sign, ll=None):
    """Posterior over a logit offset theta for p_i = sigmoid(base_i + sign*theta)."""
    return _grid_posterior(ll if ll is not None else _ll_vector(base_logits, outcomes, sign), tau)


def _coarse(post: list[float], step: int = 5):
    """Coarsen the posterior grid for expectations (keeps cost low on the live path)."""
    ts, ws = [], []
    for i in range(0, len(_THETAS), step):
        ts.append(_THETAS[i])
        ws.append(sum(post[max(0, i - step // 2): i + step // 2 + 1]))
    z = sum(ws)
    return ts, [w / z for w in ws]


def _expected_goals(base_logits: list[float], theta: float, sign: float) -> float:
    return sum(XG.sigmoid(b + sign * theta) for b in base_logits)


@dataclass
class Decomposition:
    """Observed minus expected, split into a skill part and a luck part."""
    n: int
    gate: str
    observed: float
    expected: float
    skill: Interval
    luck: Interval
    luck_share: Interval
    theta: Interval
    p_positive: float

    def as_dict(self) -> dict:
        return {
            "n": self.n, "gate": self.gate, "observed": round(self.observed, 2),
            "expected": round(self.expected, 2), "excess": round(self.observed - self.expected, 2),
            "skill": self.skill.as_dict(2), "luck": self.luck.as_dict(2),
            "luck_share": self.luck_share.as_dict(2), "theta": self.theta.as_dict(3),
            "p_skill_positive": round(self.p_positive, 2),
        }


def _share(luck: float, skill: float) -> float:
    d = abs(luck) + abs(skill)
    return 0.5 if d < 1e-9 else abs(luck) / d


def decompose_finishing(shots: Iterable[Shot], tau: float = FINISH_TAU, ll=None) -> Decomposition:
    shots = list(shots)
    n = len(shots)
    base = [XG.logit(s.xg) for s in shots]
    y = [1 if s.goal else 0 for s in shots]
    observed = float(sum(y))
    expected = sum(s.xg for s in shots)
    if n == 0:
        z = Interval(0, 0, 0)
        return Decomposition(0, "insufficient", 0, 0, z, z, Interval(0.5, 0, 1), z, 0.5)
    post = _skill_posterior(base, y, tau, +1.0, ll)
    return _finish_decomposition(post, base, n, observed, expected, +1.0, expected_sign=+1)


def decompose_keeping(faced: Iterable[Shot], tau: float = KEEPER_TAU, ll=None) -> Decomposition:
    """Goals prevented vs post-shot xG, split into keeper skill and luck."""
    faced = list(faced)
    n = len(faced)
    base = [XG.logit(s.psxg if s.psxg is not None else 0.3) for s in faced]
    y = [1 if s.goal else 0 for s in faced]
    expected_conceded = sum(s.psxg or 0.0 for s in faced)
    observed_prevented = expected_conceded - sum(y)
    if n == 0:
        z = Interval(0, 0, 0)
        return Decomposition(0, "insufficient", 0, 0, z, z, Interval(0.5, 0, 1), z, 0.5)
    # keeper skill theta lowers p(goal): p = sigmoid(base - theta)
    post = _skill_posterior(base, y, tau, -1.0, ll)
    mean_t = sum(t * w for t, w in zip(_THETAS, post))
    lo_t, hi_t = _quantile(post, 0.05), _quantile(post, 0.95)

    def prevented(theta: float) -> float:
        return expected_conceded - _expected_goals(base, theta, -1.0)

    ct, cw = _coarse(post)
    skill_mean = sum(w * prevented(t) for t, w in zip(ct, cw))
    skill = Interval(skill_mean, prevented(lo_t), prevented(hi_t))
    luck = Interval(observed_prevented - skill.mean, observed_prevented - skill.hi,
                    observed_prevented - skill.lo)
    shares = [_share(luck.lo, skill.hi), _share(luck.mean, skill.mean), _share(luck.hi, skill.lo)]
    share = Interval(shares[1], min(shares), max(shares))
    p_pos = sum(w for t, w in zip(_THETAS, post) if t > 0)
    return Decomposition(n, gate_for(n), observed_prevented, 0.0, skill, luck, share,
                         Interval(mean_t, lo_t, hi_t), p_pos)


def _finish_decomposition(post, base, n, observed, expected, sign, expected_sign):
    mean_t = sum(t * w for t, w in zip(_THETAS, post))
    lo_t, hi_t = _quantile(post, 0.05), _quantile(post, 0.95)

    def extra(theta: float) -> float:
        return _expected_goals(base, theta, sign) - expected

    ct, cw = _coarse(post)
    skill_mean = sum(w * extra(t) for t, w in zip(ct, cw))
    skill = Interval(skill_mean, extra(lo_t), extra(hi_t))
    excess = observed - expected
    luck = Interval(excess - skill.mean, excess - skill.hi, excess - skill.lo)
    shares = [_share(luck.lo, skill.hi), _share(luck.mean, skill.mean), _share(luck.hi, skill.lo)]
    share = Interval(shares[1], min(shares), max(shares))
    p_pos = sum(w for t, w in zip(_THETAS, post) if t > 0)
    return Decomposition(n, gate_for(n), observed, expected, skill, luck, share,
                         Interval(mean_t, lo_t, hi_t), p_pos)


# --- the book: history + live match --------------------------------------------------------
@dataclass
class Book:
    players: dict[str, Player]
    shots: list[Shot] = field(default_factory=list)
    tackles_won: dict[str, int] = field(default_factory=dict)
    matches: float = 0.0
    goal_value: float = GOAL_VALUE_K
    _live_frac: float = 0.0
    _cache: dict = field(default_factory=dict, repr=False)
    _ll: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_history(cls, players: dict[str, Player], history: list[Match]) -> "Book":
        b = cls(players=players)
        for m in history:
            b.shots += m.shots
            for e in m.events:
                if e["type"] == "tackle" and e.get("won"):
                    b.tackles_won[e["player"]] = b.tackles_won.get(e["player"], 0) + 1
        b.matches = float(len(history))
        for sh in b.shots:
            b._ll_add(sh)
        return b

    def _ll_add(self, sh: Shot) -> None:
        y = 1 if sh.goal else 0
        k = f"F:{sh.shooter}"
        c = _ll_contrib(XG.logit(sh.xg), y, +1.0)
        self._ll[k] = c if k not in self._ll else [a + x for a, x in zip(self._ll[k], c)]
        if sh.on_target and sh.psxg is not None:
            k = f"K:{sh.keeper}"
            c = _ll_contrib(XG.logit(sh.psxg), y, -1.0)
            self._ll[k] = c if k not in self._ll else [a + x for a, x in zip(self._ll[k], c)]

    def add_live_shot(self, shot: Shot) -> None:
        self.shots.append(shot)
        self._ll_add(shot)
        self._cache.clear()

    def set_live_fraction(self, frac: float) -> None:
        # live match counts as a fractional match so replacement-level credit is not inflated
        self._live_frac = max(0.0, min(1.0, frac))
        self._cache.clear()

    # -- per player -------------------------------------------------------------------
    def _player_shots(self, pid: str, shots: list[Shot]) -> list[Shot]:
        return [s for s in shots if s.shooter == pid]

    def row(self, pid: str, shots: list[Shot] | None = None, matches: float | None = None) -> dict:
        shots = self.shots if shots is None else shots
        matches = (self.matches + self._live_frac) if matches is None else matches
        key = (pid, len(shots), round(matches, 2), sum(1 for s in shots if s.goal))
        if shots is self.shots and key in self._cache:
            return self._cache[key]
        p = self.players[pid]
        if p.role == "GK":
            faced = [s for s in shots if s.keeper == pid and s.on_target]
            dec = decompose_keeping(faced, ll=self._ll.get(f"K:{pid}") if shots is self.shots else None)
            obs_value = dec.observed
            adj_mean, adj_lo, adj_hi = dec.skill.mean, dec.skill.lo, dec.skill.hi
        else:
            own = self._player_shots(pid, shots)
            dec = decompose_finishing(own, ll=self._ll.get(f"F:{pid}") if shots is self.shots else None)
            assisted = sum(s.xg for s in shots if s.assister == pid)
            tackles = self.tackles_won.get(pid, 0)
            side = ASSIST_WEIGHT * assisted + DEF_ACTION_VALUE * tackles
            obs_value = dec.observed + side
            adj_mean = dec.expected + dec.skill.mean + side
            adj_lo = dec.expected + dec.skill.lo + side
            adj_hi = dec.expected + dec.skill.hi + side
        repl = REPLACEMENT_PER_MATCH[p.role] * matches
        wage_k = p.wage_k_week * WEEKS_PER_YEAR * (matches / 38.0 if matches else 1.0)
        wage_k = max(wage_k, 1e-6)

        def pulse(value: float) -> float:
            return (value - repl) * self.goal_value / wage_k

        out = {
            "id": pid, "name": p.name, "name_ar": p.name_ar, "team": p.team, "role": p.role,
            "wage_k_week": p.wage_k_week, "contract_years": p.contract_years,
            "n_shots": len(self._player_shots(pid, shots)) if p.role != "GK" else dec.n,
            "decomposition": dec.as_dict(),
            "value_observed": round(obs_value, 2),
            "var_observed": obs_value - repl, "wage_k_season": wage_k,
            "pulse_observed": round(pulse(obs_value), 2),
            "pulse": {"mean": round(pulse(adj_mean), 2), "lo": round(pulse(adj_lo), 2),
                      "hi": round(pulse(adj_hi), 2)},
        }
        out["verdict"] = verdict(out)
        if shots is self.shots:
            self._cache[key] = out
        return out

    def ledger(self, roles: tuple[str, ...] = ("FWD", "MID", "DEF", "GK")) -> list[dict]:
        return [self.row(pid) for pid, p in self.players.items() if p.role in roles]

    def whatif(self, shot_id: str, goal: bool) -> dict:
        """Flip one shot's outcome and report the change for shooter and keeper."""
        target = next((s for s in self.shots if s.id == shot_id), None)
        if target is None:
            raise KeyError(shot_id)
        if goal and not target.on_target:
            raise ValueError("an off-target shot cannot be turned into a goal")
        flipped = Shot(**{**target.__dict__, "goal": goal, "saved": target.on_target and not goal})
        alt = [flipped if s.id == shot_id else s for s in self.shots]
        res = {}
        for pid in (target.shooter, target.keeper):
            before = self.row(pid)
            after = self.row(pid, shots=alt)
            res[pid] = {
                "before": _brief(before), "after": _brief(after),
                "delta_pulse_observed": round(after["pulse_observed"] - before["pulse_observed"], 2),
                "delta_pulse_evidence": round(after["pulse"]["mean"] - before["pulse"]["mean"], 2),
            }
        return {"shot_id": shot_id, "simulated_outcome": "goal" if goal else "no goal",
                "label": "simulation scenario, not what would have happened", "players": res}


def _brief(r: dict) -> dict:
    return {"pulse_observed": r["pulse_observed"], "pulse": r["pulse"], "verdict": r["verdict"],
            "luck_share": r["decomposition"]["luck_share"]}


def verdict(row: dict) -> dict:
    """Plain-language, advice-free statement derived from the evidence gate and the band."""
    gate = row["decomposition"]["gate"]
    pulse = row["pulse"]
    if gate == "insufficient":
        code = "no_evidence"
    elif pulse["lo"] > 1.0:
        code = "above"
    elif pulse["hi"] < 1.0:
        code = "below"
    else:
        code = "inconclusive"
    if code != "no_evidence" and gate == "directional" and code in ("above", "below"):
        code = code + "_directional"
    return {"code": code, "gate": gate}


def calibrate_goal_value(book: Book) -> float:
    """Set the goals-to-money constant so the median positive-value player is break-even (pulse 1.0).

    This makes the conversion a pure scale choice: only relative comparisons are meaningful."""
    ratios = sorted(r["var_observed"] / r["wage_k_season"] for r in book.ledger()
                    if r["var_observed"] > 0)
    med = ratios[len(ratios) // 2] if ratios else 0.0
    book.goal_value = (1.0 / med) if med > 0 else GOAL_VALUE_K
    book._cache.clear()
    return book.goal_value
