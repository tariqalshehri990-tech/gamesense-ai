"""Pitch geometry and a fixed broadcast-style camera (pitch plane <-> image plane homography)."""
from __future__ import annotations

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0
GOAL_W = 7.32
FPS = 25
W, H = 1280, 720

# Pitch corners (metres) -> image pixels. A raised camera on the halfway line, looking across the pitch.
PITCH_CORNERS = np.float32([[0, 0], [PITCH_L, 0], [PITCH_L, PITCH_W], [0, PITCH_W]])
IMAGE_CORNERS = np.float32([[290, 128], [990, 128], [1262, 668], [18, 668]])
H_P2I = cv2.getPerspectiveTransform(PITCH_CORNERS, IMAGE_CORNERS)
H_I2P = np.linalg.inv(H_P2I)


def to_image(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_P2I).reshape(-1, 2)


def to_pitch(pts: np.ndarray, H: np.ndarray | None = None) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H_I2P if H is None else H).reshape(-1, 2)


def px_per_metre(x: float, y: float) -> tuple[float, float]:
    """Local image scale (horizontal, vertical) in pixels per metre at pitch point (x, y)."""
    p = to_image(np.array([[x, y], [x + 1, y], [x, y + 1]]))
    return float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[2] - p[0]))


def flat_pitch(scale: int = 10) -> np.ndarray:
    """Top-down pitch texture (BGR) with mowing stripes and markings, `scale` px per metre, 4 m margin."""
    m = 4 * scale
    img = np.zeros((int(PITCH_W * scale) + 2 * m, int(PITCH_L * scale) + 2 * m, 3), np.uint8)
    for i in range(0, img.shape[1], 1):
        pass
    stripe = int(7.5 * scale)
    for k, x0 in enumerate(range(0, img.shape[1], stripe)):
        img[:, x0:x0 + stripe] = (52, 120, 46) if k % 2 == 0 else (58, 132, 52)
    c = (235, 240, 235)
    t = max(2, scale // 4)

    def P(x, y):
        return (int(x * scale) + m, int(y * scale) + m)

    cv2.rectangle(img, P(0, 0), P(PITCH_L, PITCH_W), c, t)
    cv2.line(img, P(PITCH_L / 2, 0), P(PITCH_L / 2, PITCH_W), c, t)
    cv2.circle(img, P(PITCH_L / 2, PITCH_W / 2), int(9.15 * scale), c, t)
    cv2.circle(img, P(PITCH_L / 2, PITCH_W / 2), t + 1, c, -1)
    for sx in (0, PITCH_L):
        d = 1 if sx == 0 else -1
        cv2.rectangle(img, P(sx, PITCH_W / 2 - 20.16), P(sx + d * 16.5, PITCH_W / 2 + 20.16), c, t)
        cv2.rectangle(img, P(sx, PITCH_W / 2 - 9.16), P(sx + d * 5.5, PITCH_W / 2 + 9.16), c, t)
        cv2.circle(img, P(sx + d * 11, PITCH_W / 2), t + 1, c, -1)
    return img


def render_background(scale: int = 10) -> np.ndarray:
    """Static camera view of the empty pitch (stands/ads backdrop above the far touchline)."""
    flat = flat_pitch(scale)
    m = 4 * scale
    S = np.float32([[1 / 1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # pitch metres -> flat-texture pixels
    A = np.float32([[scale, 0, m], [0, scale, m], [0, 0, 1]])
    Hf = H_P2I @ np.linalg.inv(A)
    frame = np.zeros((H, W, 3), np.uint8)
    frame[134:, :] = (44, 104, 40)
    # stands: dark gradient band with colour flecks
    rng = np.random.default_rng(3)
    for y in range(0, 150):
        frame[y, :] = (34 + y // 9, 30 + y // 10, 28 + y // 12)
    for _ in range(2200):
        x, y = int(rng.integers(0, W)), int(rng.integers(8, 120))
        cv2.circle(frame, (x, y), 2, tuple(int(v) for v in rng.integers(60, 200, 3)), -1)
    cv2.rectangle(frame, (0, 112), (W, 134), (45, 40, 36), -1)
    warped = cv2.warpPerspective(flat, Hf, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                 dst=frame.copy())
    mask = cv2.warpPerspective(np.full(flat.shape[:2], 255, np.uint8), Hf, (W, H))
    frame[mask > 0] = warped[mask > 0]
    return frame
