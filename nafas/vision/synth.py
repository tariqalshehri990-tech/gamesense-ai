"""Scripted synthetic tracking: 8 v 8, three attacks (a saved shot, a missed shot, a goal).

Everything here is generated. The generator also records ground truth (positions, passes, shots) so the
vision pipeline can be scored against it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pitch import FPS, GOAL_W, PITCH_L, PITCH_W

TEAMS = ("KBY", "RMR")
ROLES = ["GK", "DEF", "DEF", "MID", "MID", "MID", "FWD", "FWD"]
ROSTER_IDS = {"KBY": ["kby_gk", "kby_d1", "kby_d2", "kby_m1", "kby_m2", "kby_m3", "kby_f1", "kby_f2"],
              "RMR": ["rmr_gk", "rmr_d1", "rmr_d2", "rmr_m1", "rmr_m2", "rmr_m3", "rmr_f1", "rmr_f2"]}
ANCHORS = np.array([[4, 34], [24, 22], [24, 46], [46, 14], [46, 34], [46, 54], [68, 26], [68, 42]], float)
VMAX = {"GK": 5.0, "DEF": 7.2, "MID": 7.6, "FWD": 8.0}
ACC = 5.5


def team_of(i: int) -> str:
    return TEAMS[i // 8]


@dataclass
class Segment:
    kind: str                       # hold / pass / carry / shot / celebrate
    team: int                       # 0 = KBY (attacks +x), 1 = RMR (attacks -x)
    i: int = 0                      # local index of the player on the ball
    j: int = 0                      # receiver (pass)
    dur: float = 1.0
    to: tuple[float, float] = (0, 0)
    outcome: str = ""               # goal / save / miss
    aim: float = 0.0                # shot target offset from goal centre (m)
    # filled at runtime
    t0: float = 0.0
    p0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ptarget: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gk0: np.ndarray = field(default_factory=lambda: np.zeros(2))


def _dir(team: int) -> float:
    return 1.0 if team == 0 else -1.0


def anchors_for(team: int, ball: np.ndarray, attacking: bool) -> np.ndarray:
    a = ANCHORS.copy()
    if team == 1:
        a[:, 0] = PITCH_L - a[:, 0]
    shift = float(np.clip((ball[0] - PITCH_L / 2) * 0.55, -20, 20))
    push = 6.0 * _dir(team) if attacking else -2.0 * _dir(team)
    for k in range(1, 8):
        a[k, 0] = np.clip(a[k, 0] + shift + (push if k >= 3 else 0.5 * push), 6, PITCH_L - 6)
        a[k, 1] = a[k, 1] + (ball[1] - a[k, 1]) * 0.18
    a[0, 0] = 4.0 if team == 0 else PITCH_L - 4.0
    a[0, 1] = 34 + (ball[1] - 34) * 0.18
    return a


SCRIPT = [
    # Attack 1: Kestrel Bay build up, shot saved
    Segment("hold", 0, 4, dur=1.6),
    Segment("pass", 0, 4, j=5, dur=1.1),
    Segment("carry", 0, 5, to=(66, 40), dur=3.0),
    Segment("pass", 0, 5, j=7, dur=1.0),
    Segment("carry", 0, 7, to=(86, 36), dur=1.8),
    Segment("shot", 0, 7, outcome="save", aim=-1.9, dur=0.75),
    Segment("hold", 1, 0, dur=2.2),
    # Redmoor counter, shot off target
    Segment("pass", 1, 0, j=2, dur=1.4),
    Segment("pass", 1, 2, j=4, dur=1.6),
    Segment("carry", 1, 4, to=(60, 30), dur=2.4),
    Segment("pass", 1, 4, j=6, dur=1.0),
    Segment("carry", 1, 6, to=(27, 24), dur=2.6),
    Segment("shot", 1, 6, outcome="miss", aim=8.0, dur=0.9),
    Segment("hold", 0, 0, dur=2.0),
    # Attack 2: Kestrel Bay again, goal
    Segment("pass", 0, 0, j=1, dur=1.5),
    Segment("pass", 0, 1, j=4, dur=1.6),
    Segment("pass", 0, 4, j=3, dur=1.3),
    Segment("carry", 0, 3, to=(84, 12), dur=3.2),
    Segment("pass", 0, 3, j=7, dur=1.2),
    Segment("shot", 0, 7, outcome="goal", aim=2.6, dur=0.7),
    Segment("celebrate", 0, 7, dur=6.0),
]


for _s in SCRIPT:
    if _s.kind == "pass":
        _s.dur *= 1.35          # passes stay clearly slower than shots


def _smooth_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.15, 0.5, (n, 2))
    ph = rng.uniform(0, 6.28, (n, 2))
    return f, ph


@dataclass
class Clip:
    players: np.ndarray            # [T, 16, 2] metres
    ball: np.ndarray               # [T, 3] metres (x, y, height)
    truth: dict
    fps: int = FPS

    @property
    def n_frames(self) -> int:
        return self.players.shape[0]


def generate(seed: int = 5) -> Clip:
    rng = np.random.default_rng(seed)
    dt = 1.0 / FPS
    pos = np.zeros((16, 2))
    for team in (0, 1):
        pos[team * 8:(team + 1) * 8] = anchors_for(team, np.array([52.5, 34]), False)
    pos += rng.normal(0, 0.8, pos.shape)
    vel = np.zeros((16, 2))
    ball = np.array([pos[4, 0], pos[4, 1], 0.0])
    f, ph = _smooth_noise(16, seed)
    t = 0.0
    P, B = [], []
    truth = {"passes": [], "shots": [], "possession": [], "fps": FPS}
    seg_i = 0
    segs = [Segment(**{k: v for k, v in s.__dict__.items()}) for s in SCRIPT]
    segs[0].t0 = 0.0
    started = False
    ball_xy = ball[:2].copy()
    end_t = sum(s.dur for s in segs) + 1.0
    holder = (0, 4)
    while t < end_t:
        if seg_i < len(segs):
            s = segs[seg_i]
            if not started:
                s.t0 = t
                gi = s.team * 8 + s.i
                s.p0 = ball_xy.copy()
                if s.kind == "pass":
                    gj = s.team * 8 + s.j
                    lead = vel[gj] * 0.5
                    s.ptarget = np.clip(pos[gj] + lead + np.array([_dir(s.team) * 2.0, 0]), 2, [PITCH_L - 2, PITCH_W - 2])
                    truth["passes"].append({"t": t, "team": TEAMS[s.team], "from": s.i, "to": s.j})
                elif s.kind == "carry":
                    s.ptarget = np.array(s.to, float)
                elif s.kind == "shot":
                    gx = PITCH_L if s.team == 0 else 0.0
                    s.ptarget = np.array([gx, PITCH_W / 2 + s.aim])
                    s.gk0 = pos[(1 - s.team) * 8].copy()
                    d = math.hypot(gx - ball_xy[0], PITCH_W / 2 - ball_xy[1])
                    truth["shots"].append({"t": t, "team": TEAMS[s.team], "player": s.i, "x": float(ball_xy[0]),
                                           "y": float(ball_xy[1]), "outcome": s.outcome, "distance": d,
                                           "aim": s.aim})
                started = True
            u = min(1.0, (t - s.t0) / max(s.dur, 1e-6))
            targets = {}
            forced = {}
            att = s.team
            if s.kind == "hold":
                gi = s.team * 8 + s.i
                ball_xy = pos[gi] + np.array([_dir(s.team) * 0.5, 0.0])
                ball[2] = 0.0
                holder = (s.team, s.i)
            elif s.kind == "pass":
                ball_xy = s.p0 + (s.ptarget - s.p0) * u
                ball[2] = 0.18 * np.linalg.norm(s.ptarget - s.p0) * math.sin(math.pi * u) ** 1.0 * 0.35
                gj = s.team * 8 + s.j
                targets[gj] = (s.ptarget, VMAX[ROLES[s.j]])
                if u >= 1.0:
                    holder = (s.team, s.j)
            elif s.kind == "carry":
                gi = s.team * 8 + s.i
                targets[gi] = (s.ptarget, 5.4)
                ball_xy = pos[gi] + np.array([_dir(s.team) * 0.5, 0.0])
                ball[2] = 0.0
                holder = (s.team, s.i)
            elif s.kind == "shot":
                ball_xy = s.p0 + (s.ptarget - s.p0) * u
                ball[2] = 0.9 * math.sin(math.pi * u) + 0.3
                opp = 1 - s.team
                gk = opp * 8
                if s.outcome == "save":
                    dest = s.ptarget + np.array([-_dir(s.team) * 1.4, 0.0])
                elif s.outcome == "goal":      # keeper beaten: dives late and the wrong way
                    dest = s.ptarget + np.array([-_dir(s.team) * 1.0, -math.copysign(3.4, s.aim)])
                else:
                    dest = np.array([s.ptarget[0] - _dir(s.team) * 2.0, PITCH_W / 2 + s.aim * 0.2])
                ease = u * u * (3 - 2 * u) if s.outcome == "save" else (u ** 2.5 if s.outcome == "goal" else u * u * (3 - 2 * u))
                forced[gk] = s.gk0 + (dest - s.gk0) * ease
                if u >= 1.0:
                    if s.outcome == "save":
                        holder = (opp, 0)
                    elif s.outcome == "goal":
                        ball_xy = np.array([s.ptarget[0] + _dir(s.team) * 1.0, s.ptarget[1]])
                        ball[2] = 0.4
            elif s.kind == "celebrate":
                shooter = s.team * 8 + s.i
                for k in range(8):
                    g = s.team * 8 + k
                    if g != shooter:
                        targets[g] = (pos[shooter] + rng.normal(0, 3.0, 2) * 0 + np.array([-_dir(s.team) * (2 + k % 3), (k - 4) * 0.8]), 6.5)
                targets[shooter] = (np.array([PITCH_L - 18 if s.team == 0 else 18, 62.0]), 6.0)
                ball_xy = np.array([PITCH_L - 1.0 if s.team == 0 else 1.0, PITCH_W / 2 + s.aim])
                ball[2] = 0.0
            if (t - s.t0) >= s.dur and not (s.kind == "pass" and u < 1.0):
                seg_i += 1
                started = False
        else:
            targets = {}
            forced = {}
        # ---- opponent pressure + formation targets ---------------------------------------
        poss_team = holder[0]
        for team in (0, 1):
            anc = anchors_for(team, ball_xy, team == poss_team)
            for k in range(8):
                g = team * 8 + k
                if g in targets:
                    continue
                tgt = anc[k] + np.array([math.sin(f[g, 0] * t + ph[g, 0]), math.sin(f[g, 1] * t + ph[g, 1])]) * 1.6
                targets[g] = (tgt, VMAX[ROLES[k]] * 0.7)
        defenders = [(np.linalg.norm(pos[g] - ball_xy), g) for g in range((1 - poss_team) * 8 + 1, (1 - poss_team) * 8 + 8)]
        defenders.sort()
        if seg_i < len(segs) and segs[seg_i].kind in ("carry", "pass", "hold") and not (ball_xy[0] < 4 or ball_xy[0] > PITCH_L - 4):
            for _, g in defenders[:2]:
                if g not in [x for x in ()]:
                    targets[g] = (ball_xy + np.array([-_dir(poss_team) * 1.2, 0.0]), 6.2)
        # ---- integrate ------------------------------------------------------------------------
        for g in range(16):
            tgt, vmax = targets[g]
            want = (tgt - pos[g]) / 0.6
            sp = np.linalg.norm(want)
            if sp > vmax:
                want *= vmax / sp
            dv = want - vel[g]
            n = np.linalg.norm(dv)
            if n > ACC * dt:
                dv *= ACC * dt / n
            vel[g] += dv
            pos[g] += vel[g] * dt
        for g, p_ in forced.items():
            pos[g] = p_
            vel[g] = 0.0
        pos[:, 0] = np.clip(pos[:, 0], 0.5, PITCH_L - 0.5)
        pos[:, 1] = np.clip(pos[:, 1], 0.5, PITCH_W - 0.5)
        P.append(pos.copy())
        B.append([ball_xy[0], ball_xy[1], ball[2]])
        t += dt
    truth["duration"] = len(P) / FPS
    truth["holder_final"] = holder
    return Clip(players=np.array(P), ball=np.array(B), truth=truth)


if __name__ == "__main__":
    c = generate()
    print(c.players.shape, c.truth["duration"], len(c.truth["shots"]))
