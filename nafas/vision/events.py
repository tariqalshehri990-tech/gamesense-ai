"""Tracking -> physical metrics -> football events (possession, passes, pressure, shots with outcomes)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pitch import GOAL_W, PITCH_L, PITCH_W
from .track import Tracks

POSSESSION_RADIUS = 2.4
SHOT_SPEED = 15.5          # m/s ball speed that marks a shot candidate


def smooth(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    pad = k // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    ker = np.ones(k) / k
    out = np.empty_like(a, dtype=float)
    for j in np.ndindex(*a.shape[1:]):
        out[(slice(None),) + j] = np.convolve(ap[(slice(None),) + j], ker, mode="valid")
    return out


def player_metrics(tr: Tracks) -> dict:
    fps = tr.fps
    sm = smooth(tr.players, 5)
    v = np.gradient(sm, axis=0) * fps
    speed = smooth(np.linalg.norm(v, axis=2), 7)           # m/s, [T,16]
    dist = np.nansum(np.linalg.norm(np.diff(sm, axis=0), axis=2), axis=0)
    sprint = (speed > 7.0)
    sprints = [int(np.sum(np.diff(sprint[:, i].astype(int)) == 1)) for i in range(16)]
    out = {}
    for i, pid in enumerate(tr.ids):
        out[pid] = {"top_speed_kmh": round(float(np.nanmax(speed[:, i]) * 3.6), 1),
                    "distance_m": round(float(dist[i]), 1), "sprints": sprints[i]}
    return {"players": out, "speed": speed}


def team_shape(tr: Tracks, t: int, team: int) -> dict:
    pts = tr.players[t, team * 8 + 1:(team + 1) * 8]            # outfield
    return {"centroid_x": round(float(pts[:, 0].mean()), 1), "width": round(float(np.ptp(pts[:, 1])), 1),
            "length": round(float(np.ptp(pts[:, 0])), 1)}


@dataclass
class Event:
    kind: str
    t: float
    team: str
    player: str
    detail: dict


def derive_events(tr: Tracks) -> list[Event]:
    fps = tr.fps
    T = len(tr.ball)
    ball = smooth(tr.ball, 3)
    bv = np.gradient(ball, axis=0) * fps
    bspeed = np.linalg.norm(bv, axis=1)
    teams = ["KBY", "RMR"]
    ev: list[Event] = []
    # --- possession with hysteresis ---------------------------------------------------------
    owner = np.full(T, -1)
    cur = -1
    for t in range(T):
        d = np.linalg.norm(tr.players[t] - ball[t], axis=1)
        j = int(np.argmin(d))
        if d[j] < POSSESSION_RADIUS and bspeed[t] < 12.0:
            if cur != j and (cur < 0 or d[j] < np.linalg.norm(tr.players[t, cur] - ball[t]) - 0.4):
                cur = j
        elif cur >= 0 and np.linalg.norm(tr.players[t, cur] - ball[t]) > POSSESSION_RADIUS * 2.2:
            cur = -1
        owner[t] = cur
    # --- passes: owner changes inside a team ------------------------------------------------
    last_owner, last_t = -1, 0
    for t in range(T):
        if owner[t] >= 0:
            if last_owner >= 0 and owner[t] != last_owner and (t - last_t) / fps < 3.5:
                a, b = last_owner, owner[t]
                if a // 8 == b // 8 and np.linalg.norm(ball[t] - ball[last_t]) > 4.0:
                    ev.append(Event("pass", last_t / fps, teams[a // 8], tr.ids[a],
                                    {"to": tr.ids[b], "length_m": round(float(np.linalg.norm(tr.players[t, b] - tr.players[last_t, a])), 1),
                                     "t_received": round(t / fps, 2)}))
            last_owner, last_t = int(owner[t]), t
    # --- shots: fast ball leaving a holder toward the opponent goal -------------------------
    t = 1
    while t < T - 5:
        if bspeed[t] > SHOT_SPEED and bspeed[t - 1] <= bspeed[t]:
            # the kicker is the player closest to the ball around the moment it accelerates
            shooter, best = -1, 2.6
            for k in range(max(t - 8, 0), min(t + 2, T)):
                dk = np.linalg.norm(tr.players[k] - ball[k], axis=1)
                jk = int(np.argmin(dk))
                if dk[jk] < best:
                    shooter, best = jk, float(dk[jk])
            if shooter >= 0:
                team = shooter // 8
                gx = PITCH_L if team == 0 else 0.0
                end = min(T - 1, t + int(2.5 * fps))
                # direction from a short regression over the first 0.4 s of flight (robust to detector jitter)
                k = min(T - 1, t + 10)
                tt = np.arange(t, k + 1) / fps
                vx = float(np.polyfit(tt, ball[t:k + 1, 0], 1)[0])
                vy = float(np.polyfit(tt, ball[t:k + 1, 1], 1)[0])
                sp = math.hypot(vx, vy)
                x0, y0 = tr.players[t, shooter]
                dist = math.hypot(gx - x0, PITCH_W / 2 - y0)
                if (gx - ball[t, 0]) * vx > 0 and abs(vx) > 0.6 * sp and dist < 36:
                    opp_gk = (1 - team) * 8
                    mates = [m for m in range(team * 8 + 1, team * 8 + 8) if m != shooter]
                    received = False
                    for kk in range(t + 5, end):
                        if any(np.linalg.norm(tr.players[kk, m] - ball[kk]) < 1.6 for m in mates) and bspeed[kk] < 9:
                            received = True
                            break
                    if not received:
                        sign = 1 if team == 0 else -1
                        cross = np.where((ball[t:end, 0] - gx) * sign >= -0.4)[0]
                        kd_all = np.linalg.norm(tr.players[t:end, opp_gk] - ball[t:end], axis=1)
                        if len(cross):
                            ci = cross[0]
                            y_cross = float(ball[t + ci, 1])
                            in_mouth = abs(y_cross - PITCH_W / 2) <= GOAL_W / 2 + 1.0
                            if not in_mouth:
                                outcome = "off_target"
                            elif kd_all[ci] < 2.8:
                                outcome = "save"
                            else:
                                outcome = "goal"
                        else:
                            y_cross = float(ball[end, 1])
                            outcome = "save" if kd_all.min() < 2.4 else "off_target"
                        a1 = math.atan2(y0 - (PITCH_W / 2 - GOAL_W / 2), abs(gx - x0))
                        a2 = math.atan2(y0 - (PITCH_W / 2 + GOAL_W / 2), abs(gx - x0))
                        opp = tr.players[t, (1 - team) * 8 + 1:(1 - team) * 8 + 8]
                        close = int(np.sum(np.linalg.norm(opp - tr.players[t, shooter], axis=1) < 3.5))
                        ev.append(Event("shot", t / fps, teams[team], tr.ids[shooter],
                                        {"x": round(float(x0), 1), "y": round(float(y0), 1), "distance": round(dist, 1),
                                         "angle": round(abs(a1 - a2), 3), "speed_ms": round(float(bspeed[t]), 1),
                                         "aim_y": round(y_cross, 1), "outcome": outcome, "pressure": min(close, 2)}))
                        t += int(2.0 * fps)
                        continue
        t += 1
    ev.sort(key=lambda e: e.t)
    return ev
