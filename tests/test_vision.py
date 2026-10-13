"""Vision layer: geometry, calibration, event→Tariq bridge and HUD helpers (no video decode needed)."""
import numpy as np

from nafas.vision import bridge, compose, pitch
from nafas.vision.events import Event
from nafas.vision.track import calibrate, to_metres


def test_homography_roundtrip():
    pts = np.array([[10.0, 10.0], [52.5, 34.0], [100.0, 60.0]])
    back = pitch.to_pitch(pitch.to_image(pts))
    assert np.abs(back - pts).max() < 0.01


def test_calibration_from_landmarks_recovers_metres():
    from nafas.vision.pipeline import LANDMARKS_M
    lm = np.asarray(LANDMARKS_M, dtype=np.float32)
    px = pitch.to_image(lm) + np.random.default_rng(0).normal(0, 0.7, lm.shape)
    Hm = calibrate(px, lm)
    probe = np.array([[30.0, 20.0], [80.0, 50.0]])
    assert np.abs(to_metres(Hm, pitch.to_image(probe)) - probe).max() < 0.6


def _events():
    return [Event(kind="shot", t=9.6, team="KBY", player="kby_f2",
                  detail=dict(x=86.2, y=36.1, distance=19.0, angle=0.38, speed_ms=15.8, aim_y=30.8, outcome="save", pressure=0)),
            Event(kind="shot", t=37.5, team="KBY", player="kby_f2",
                  detail=dict(x=88.0, y=33.0, distance=17.0, angle=0.42, speed_ms=17.0, aim_y=32.0, outcome="goal", pressure=0))]


def test_bridge_builds_nafas_shots():
    f = bridge.facts_from_video(35, _events(), compose.T0)
    shots = f.match.shots
    assert len(shots) == 2
    assert shots[0].saved and not shots[0].goal
    assert shots[1].goal
    assert all(0.01 < s.xg < 0.9 for s in shots)


def test_hud_helpers():
    assert compose.clock(0) == "23:10"
    assert compose.ltr("1").startswith("⁦")
