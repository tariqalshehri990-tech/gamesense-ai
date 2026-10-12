"""End-to-end: synthetic clip -> video -> tracking -> events -> evaluation against ground truth."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from . import events as EV
from . import pitch, synth, track

# Pitch-line landmarks an annotator (or a line-detector) would mark: corners, halfway line, penalty-box corners.
LANDMARKS_M = np.array([[0, 0], [105, 0], [105, 68], [0, 68], [52.5, 0], [52.5, 68], [0, 13.84], [16.5, 13.84],
                        [16.5, 54.16], [0, 54.16], [105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]], float)


def calibration(seed: int = 1, px_error: float = 1.0):
    """Homography from landmarks marked with ~1 px of annotation error; returns (H, held-out error in metres)."""
    rng = np.random.default_rng(seed)
    px = pitch.to_image(LANDMARKS_M) + rng.normal(0, px_error, (len(LANDMARKS_M), 2))
    Hm = track.calibrate(px, LANDMARKS_M)
    probe = np.array([[52.5 + 9.15, 34], [30, 20], [75, 50], [100, 5], [10, 60]], float)       # not used for fitting
    back = track.to_metres(Hm, pitch.to_image(probe))
    return Hm, float(np.mean(np.linalg.norm(back - probe, axis=1)))


def evaluate(tr: track.Tracks, clip: synth.Clip, evs: list[EV.Event]) -> dict:
    n = len(tr.players)
    err = np.linalg.norm(tr.players - clip.players[:n], axis=2)
    be = np.linalg.norm(tr.ball - clip.ball[:n, :2], axis=1)
    sp_true = np.linalg.norm(np.gradient(EV.smooth(clip.players[:n], 5), axis=0) * clip.fps, axis=2)
    sp_est = EV.player_metrics(tr)["speed"]
    lost = float(np.nanmean(err > 2.0))            # share of player-frames where the track is >2 m from its player
    truth_shots = clip.truth["shots"]
    det_shots = [e for e in evs if e.kind == "shot"]
    match = 0
    outcome_ok = 0
    omap = {"goal": "goal", "save": "save", "miss": "off_target"}
    for s in truth_shots:
        c = [d for d in det_shots if d.team == s["team"] and abs(d.t - s["t"]) < 1.0]
        if c:
            match += 1
            outcome_ok += c[0].detail["outcome"] == omap[s["outcome"]]
    det_pass = [e for e in evs if e.kind == "pass"]
    tp = sum(any(p["team"] == d.team and abs(p["t"] - d.t) < 1.2 for p in clip.truth["passes"]) for d in det_pass)
    pr = sum(any(p["team"] == d.team and abs(p["t"] - d.t) < 1.2 for d in det_pass) for p in clip.truth["passes"])
    return {
        "player_position_rmse_m": round(float(np.sqrt(np.nanmean(err ** 2))), 3),
        "player_position_p95_m": round(float(np.nanpercentile(err, 95)), 3),
        "ball_median_error_m": round(float(np.nanmedian(be)), 2),
        "ball_detected_share": round(float(tr.ball_seen.mean()), 3),
        "speed_mae_ms": round(float(np.nanmean(np.abs(sp_est - sp_true))), 3),
        "player_frames_off_by_2m_share": round(lost, 4),
        "shots_true": len(truth_shots), "shots_found": len(det_shots), "shots_matched": match,
        "shot_outcomes_correct": outcome_ok,
        "false_shots": len(det_shots) - match,
        "passes_true": len(clip.truth["passes"]), "passes_found": len(det_pass),
        "pass_precision": round(tp / max(1, len(det_pass)), 2), "pass_recall": round(pr / max(1, len(clip.truth["passes"])), 2),
    }


def anchors() -> np.ndarray:
    base = np.array([52.5, 34])
    return np.vstack([synth.anchors_for(0, base, False), synth.anchors_for(1, base, False)])


def analyse(video: str, cache: str | None = None):
    Hm, cal_err = calibration()
    det = None
    if cache and Path(cache).exists():
        det = pickle.load(open(cache, "rb"))
    elif cache:
        det = track.detect_video(video)
        pickle.dump(det, open(cache, "wb"))
    tr = track.run(video, Hm, synth.ROSTER_IDS, anchors(), detections=det)
    tr.calibration = {"heldout_error_m": round(cal_err, 3)}
    return tr, EV.derive_events(tr), Hm
