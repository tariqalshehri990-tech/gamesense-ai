"""Video -> tracking data: detect, track, calibrate, measure, derive match events.

Detector: background subtraction (static camera) + kit-colour segmentation, behind a small `Detection`
interface. For a panning broadcast feed, swap `detect_frame` for a YOLO person/ball detector plus camera-motion
compensation; everything downstream (tracking, metrics, events) is detector-agnostic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .pitch import FPS, GOAL_W, PITCH_L, PITCH_W

# Kit colours (BGR) come from match metadata: team A shirt, team B shirt, keeper A, keeper B.
KITS = {"KBY": (222, 120, 18), "RMR": (40, 50, 225), "KBY_GK": (30, 215, 235), "RMR_GK": (200, 60, 190)}
GROUP_ORDER = ["KBY", "RMR", "KBY_GK", "RMR_GK"]


@dataclass
class Detection:
    group: str
    foot_px: tuple[float, float]
    area: float


def median_background(path: str, samples: int = 70) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def detect_frame(frame: np.ndarray, bg: np.ndarray) -> tuple[list[Detection], list[tuple[float, float, float]]]:
    diff = cv2.absdiff(frame, bg).max(axis=2)
    fg = (diff > 30).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dets: list[Detection] = []
    ys, xs = np.nonzero(fg)                       # colour tests only on the (sparse) foreground pixels
    cols = frame[ys, xs].astype(np.int32)
    for group in GROUP_ORDER:
        d = np.linalg.norm(cols - np.array(KITS[group], np.int32), axis=1)
        sel = d < 62
        if not sel.any():
            continue
        m = np.zeros(fg.shape, np.uint8)
        m[ys[sel], xs[sel]] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if a < 5:
                continue
            # shirt blob -> feet: shirt is 0.42 of body height, shorts+legs are 0.38 below it
            foot_y = y + h + 0.9 * h
            dets.append(Detection(group, (x + w / 2, foot_y), float(a)))
    # ball: small, bright, moving foreground not part of a kit blob
    bright = ((frame.min(axis=2) > 205) & (fg > 0)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    balls = [(float(cents[k][0]), float(cents[k][1]), float(stats[k][4])) for k in range(1, n) if 2 <= stats[k][4] <= 60]
    return dets, balls


class Kalman:
    def __init__(self, xy: np.ndarray, q: float = 4.0, r: float = 0.6):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 25.0, 25.0])
        self.q, self.r = q, r
        self.misses = 0

    def predict(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q ** 2
        return self.x[:2]

    def update(self, z: np.ndarray) -> None:
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = Hm @ self.P @ Hm.T + np.eye(2) * self.r ** 2
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P
        self.misses = 0


@dataclass
class Tracks:
    fps: int
    players: np.ndarray                 # [T, 16, 2] metres, NaN when unseen
    ball: np.ndarray                    # [T, 2]
    ball_seen: np.ndarray               # [T] bool
    ids: list[str]                      # roster id for each of the 16 slots
    calibration: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def calibrate(landmarks_px: np.ndarray, landmarks_m: np.ndarray) -> np.ndarray:
    """Homography image -> pitch from pitch-line landmark correspondences (RANSAC)."""
    Hm, _ = cv2.findHomography(landmarks_px.astype(np.float32), landmarks_m.astype(np.float32), cv2.RANSAC, 3.0)
    return Hm


def to_metres(Hm: np.ndarray, px: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(px, np.float32).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def detect_video(video: str, progress=None) -> tuple[float, list]:
    """Run the detector over every frame. Returns (fps, [(detections, ball_candidates), ...])."""
    bg = median_background(video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(detect_frame(frame, bg))
        if progress and len(out) % 200 == 0:
            progress(len(out))
    cap.release()
    return fps, out


def run(video: str, Hm: np.ndarray, roster: dict[str, list[str]], anchors: np.ndarray, progress=None,
        detections: tuple[float, list] | None = None) -> Tracks:
    fps, frames = detections or detect_video(video, progress)
    return associate(fps, frames, Hm, roster, anchors)


def associate(fps: float, frames: list, Hm: np.ndarray, roster: dict[str, list[str]], anchors: np.ndarray) -> Tracks:
    dt = 1.0 / fps
    n_frames = len(frames)
    slots = {"KBY": list(range(0, 8)), "RMR": list(range(8, 16))}
    # group -> track slots. Keepers are slot 0 / 8; outfield are 1..7 / 9..15
    group_slots = {"KBY": slots["KBY"][1:], "RMR": slots["RMR"][1:], "KBY_GK": [0], "RMR_GK": [8]}
    kf: dict[int, Kalman] = {}
    players = np.full((n_frames, 16, 2), np.nan)
    ball_xy = np.full((n_frames, 2), np.nan)
    ball_seen = np.zeros(n_frames, bool)
    ball_kf: Kalman | None = None
    t = 0
    for dets, balls in frames:
        for group, sl in group_slots.items():
            gd = [d for d in dets if d.group == group]
            pts = to_metres(Hm, np.array([d.foot_px for d in gd])) if gd else np.zeros((0, 2))
            if not kf or not all(s in kf for s in sl):
                # initialise: assign detections to slots by formation anchors (kick-off shape)
                if len(gd) >= len(sl):
                    team = 0 if group.startswith("KBY") else 1
                    anc = anchors[np.array(sl)]
                    cost = np.linalg.norm(pts[:, None, :] - anc[None, :, :], axis=2)
                    r, c = linear_sum_assignment(cost)
                    for ri, ci in zip(r, c):
                        kf[sl[ci]] = Kalman(pts[ri])
                continue
            preds = np.array([kf[s].predict(dt) for s in sl])
            if len(pts):
                cost = np.linalg.norm(preds[:, None, :] - pts[None, :, :], axis=2)
                r, c = linear_sum_assignment(cost)
                used = set()
                for ri, ci in zip(r, c):
                    if cost[ri, ci] < 5.0:
                        kf[sl[ri]].update(pts[ci])
                        used.add(ri)
                for ri in range(len(sl)):
                    if ri not in used:
                        kf[sl[ri]].misses += 1
            else:
                for s in sl:
                    kf[s].misses += 1
        for s, k in kf.items():
            players[t, s] = k.x[:2]
        # ball: gated nearest-candidate association; the gate widens while the ball is unseen
        bp = to_metres(Hm, np.array([[b[0], b[1]] for b in balls])) if balls else np.zeros((0, 2))
        if ball_kf is None:
            if len(bp):
                j = int(np.argmin(np.linalg.norm(bp - np.array([52.5, 34]), axis=1)))
                if np.linalg.norm(bp[j] - np.array([52.5, 34])) < 25:
                    ball_kf = Kalman(bp[j], q=18.0, r=0.8)
                    ball_seen[t] = True
        else:
            pred = ball_kf.predict(dt)
            gate = min(30.0, 4.0 + 1.5 * ball_kf.misses)
            if len(bp):
                d = np.linalg.norm(bp - pred, axis=1)
                j = int(np.argmin(d))
                if d[j] < gate:
                    ball_kf.update(bp[j])
                    ball_seen[t] = True
                else:
                    ball_kf.misses += 1
            else:
                ball_kf.misses += 1
            if ball_kf.misses:
                ball_kf.x[2:] *= 0.85                 # do not let the prediction run away
        if ball_kf is not None:
            ball_xy[t] = ball_kf.x[:2]
        t += 1
    ids = roster["KBY"] + roster["RMR"]
    return Tracks(fps=int(round(fps)), players=players[:t], ball=ball_xy[:t], ball_seen=ball_seen[:t], ids=ids)
