"""Render a tracking clip as broadcast-style video (static wide camera)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .pitch import FPS, H, W, px_per_metre, render_background, to_image
from .synth import Clip

# BGR shirt colours. Keepers get their own colours so the detector can tell all four groups apart.
SHIRT = {0: (222, 120, 18), 1: (40, 50, 225)}
KEEPER = {0: (30, 215, 235), 1: (200, 60, 190)}
SHORTS = (28, 28, 34)
SKIN = (140, 175, 215)


def draw_frame(bg: np.ndarray, players: np.ndarray, ball: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    img = bg.copy()
    foot = to_image(players)
    order = np.argsort(foot[:, 1])
    for g in order:
        team, k = divmod(int(g), 8)
        x, y = foot[g]
        hs, _ = px_per_metre(players[g, 0], players[g, 1])
        h = 2.5 * hs
        cv2.ellipse(img, (int(x), int(y)), (int(0.45 * hs) + 2, int(0.16 * hs) + 1), 0, 0, 360, (30, 70, 28), -1)
        col = KEEPER[team] if k == 0 else SHIRT[team]
        w = max(5, int(0.62 * hs))
        cv2.rectangle(img, (int(x - w / 2), int(y - 0.38 * h)), (int(x + w / 2), int(y)), SHORTS, -1)       # shorts + legs
        cv2.rectangle(img, (int(x - w / 2), int(y - 0.80 * h)), (int(x + w / 2), int(y - 0.38 * h)), col, -1)  # shirt
        cv2.circle(img, (int(x), int(y - 0.90 * h)), max(2, int(0.13 * h)), SKIN, -1)
    bx, by = to_image(ball[None, :2])[0]
    hs, _ = px_per_metre(ball[0], ball[1])
    cv2.ellipse(img, (int(bx), int(by)), (4, 2), 0, 0, 360, (30, 70, 28), -1)
    cv2.circle(img, (int(bx), int(by - ball[2] * hs * 1.1 - 3)), 3, (245, 245, 245), -1)
    if rng is not None:
        img = cv2.add(img, rng.normal(0, 2.5, img.shape).astype(np.int16).clip(0, 255).astype(np.uint8)) if False else img
    return img


def render_video(clip: Clip, out: str | Path, crf: int = 17) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bg = render_background()
    rng = np.random.default_rng(11)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(clip.n_frames):
        f = draw_frame(bg, clip.players[t], clip.ball[t])
        noise = rng.normal(0, 2.0, f.shape)
        f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    return out


if __name__ == "__main__":
    from .synth import generate
    c = generate()
    render_video(c, "/tmp/claude-0/shots/raw.mp4")
    cap = cv2.VideoCapture("/tmp/claude-0/shots/raw.mp4")
    for idx in (80, 330, 480):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        cv2.imwrite(f"/tmp/claude-0/shots/raw_{idx}.png", fr)
    print("ok")
