"""Shot-quality models used by both the simulator and the analytics layer.

Everything here is a *synthetic-world* model. The simulator draws shots from it and the
analytics layer scores shots with the same functions. That circularity is deliberate and
disclosed in the README: the point is a transparent, checkable pipeline, not a claim about
real football.

Calibration targets (public Premier League figures, see README):
  * ~13 shots per team per match
  * ~0.115 xG per shot, ~11-12% conversion
  * ~1/3 of shots on target
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- shot-quality (pre-shot) ------------------------------------------------------------
B0, B_LOGDIST, B_ANGLE = 3.30, -1.42, 1.15
HEAD_PENALTY = -0.55
PRESSURE_PENALTY = (0.0, -0.30, -0.65)  # pressure level 0 / 1 / 2
ASSIST_BONUS = {"none": 0.0, "pass": 0.10, "cross": -0.10, "through": 0.45, "set_piece": -0.15}

# --- on-target and post-shot models ----------------------------------------------------
O0, O1 = -0.55, 0.22        # P(on target) = sigmoid(O0 + O1 * s)
G0, G1, G2 = -0.70, 0.60, 2.30  # P(goal | on target, placement d) = sigmoid(G0 + G1*s + G2*(d-0.5))
PENALTY_P_OT, PENALTY_PSXG = 0.92, 0.86

# Quadrature over placement difficulty d ~ Beta(2,2) (density 6 d (1-d)).
_GRID_D = [(i + 0.5) / 9 for i in range(9)]
_W = [6 * d * (1 - d) for d in _GRID_D]
_WS = sum(_W)
_W = [w / _WS for w in _W]


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@dataclass(frozen=True)
class ShotFeatures:
    distance: float          # metres from goal centre
    angle: float             # radians, width of goal as seen from the shot
    head: bool = False
    pressure: int = 0        # 0, 1, 2
    assist: str = "none"     # none / pass / cross / through / set_piece
    penalty: bool = False


def shot_quality(f: ShotFeatures) -> float:
    """Pre-shot quality `s` in logit space (before on-target / placement)."""
    return (
        B0
        + B_LOGDIST * math.log(max(f.distance, 1.0))
        + B_ANGLE * f.angle
        + (HEAD_PENALTY if f.head else 0.0)
        + PRESSURE_PENALTY[min(max(f.pressure, 0), 2)]
        + ASSIST_BONUS.get(f.assist, 0.0)
    )


def p_on_target(f: ShotFeatures) -> float:
    if f.penalty:
        return PENALTY_P_OT
    return sigmoid(O0 + O1 * shot_quality(f))


def psxg(f: ShotFeatures, d: float) -> float:
    """Post-shot xG: chance a shot that is on target beats an average keeper.

    `d` is placement difficulty for the keeper in [0, 1] (1 = into the corner)."""
    if f.penalty:
        return PENALTY_PSXG
    return sigmoid(G0 + G1 * shot_quality(f) + G2 * (d - 0.5))


def xg(f: ShotFeatures) -> float:
    """Pre-shot expected goals for an average shooter against an average keeper."""
    if f.penalty:
        return PENALTY_P_OT * PENALTY_PSXG
    s = shot_quality(f)
    pot = sigmoid(O0 + O1 * s)
    return pot * sum(w * sigmoid(G0 + G1 * s + G2 * (d - 0.5)) for w, d in zip(_W, _GRID_D))
