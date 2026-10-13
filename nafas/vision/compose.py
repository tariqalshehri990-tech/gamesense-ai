"""Compose the broadcast video: tracked match footage + Tariq commentary, ledger, metrics and radar graphics.

    PYTHONPATH=. python -m nafas.vision.compose raw.mp4 out.mp4 [--cache dets.pkl] [--no-audio]

Everything on screen is derived from the video itself: tracking rings, pass lines, shot cones and the radar come
from the tracker's own output (not from the generator's ground truth); the words come from the Tariq session built on
the video-derived events.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

from . import bridge, pipeline
from .hud import Hud, state_js
from .pitch import GOAL_W, PITCH_L, PITCH_W

OW, OH = 1920, 1080
SC = OW / 1280
T0 = 23 * 60 + 10
SEED = 35
TEAM_BGR = {0: (224, 123, 46), 1: (45, 57, 229)}           # KBY blue, RMR red (BGR)
MINT, GOLD, FLARE = (160, 214, 79), (58, 182, 245), (74, 107, 255)
RADAR = (1528, 664, 320, 207)                                # x, y, w, h of the pitch inside the radar panel
LANE_X0, LANE_W = 73.5, 1824 - 48 - 170
OUT = {"goal": ("هدف", "#F5B63A"), "save": ("تصدي", "#4FD6A0"), "off": ("خارج المرمى", "#FF6B4A"),
       "miss": ("خارج المرمى", "#FF6B4A"), "block": ("محجوبة", "#A9C4B3")}


def ltr(s: str) -> str:
    return "⁦" + s + "⁩"


def clock(t: float) -> str:
    s = int(T0 + t)
    return f"{s // 60}:{s % 60:02d}"


def build_sessions(evs):
    from nafas.agents import build_session
    facts = bridge.facts_from_video(SEED, evs, T0)
    ar = asyncio.run(build_session(SEED, lang="ar", facts=facts))
    en = asyncio.run(build_session(SEED, lang="en", facts=facts))
    return facts, ar, en


def window(ss, profile, dur):
    return [u for u in ss["profiles"][profile]["utterances"] if 0 <= u["start"] - T0 <= dur]


class Story:
    """Everything the HUD needs, as a pure function of time."""

    def __init__(self, tr, evs, facts, ar, en, dur, profile="investor"):
        self.tr, self.evs, self.facts, self.dur, self.profile = tr, evs, facts, dur, profile
        self.ar, self.en = ar, en
        self.names = {p["id"]: p for p in ar["meta"]["players"]}
        self.teams = {k: v["name_ar"] for k, v in ar["meta"]["teams"].items()}
        self.utt = window(ar, profile, dur)
        en_by = {u["moment_id"]: u["text"] for u in window(en, profile, dur)}
        for u in self.utt:
            u["en"] = en_by.get(u["moment_id"], "")
        self.shots = [e for e in evs if e.kind == "shot"]
        self.xg = {i: s.xg for i, s in enumerate(facts.match.shots)}
        from .events import player_metrics
        pm = player_metrics(tr)
        self.speed = pm["speed"]
        sm = np.nan_to_num(tr.players)
        self.cum = np.concatenate([np.zeros((1, 16)), np.cumsum(np.linalg.norm(np.diff(sm, axis=0), axis=2), axis=0)])
        self.timeline = [(l["t"] - T0, l) for l in ar["ledger_timeline"]]
        self.goals = [e.t + 1.2 for e in self.shots if e.detail.get("outcome") == "goal"]
        self.carrier: tuple[int, float] | None = None

    def frame_of(self, t):
        return min(int(t * self.tr.fps), len(self.tr.ball) - 1)

    def possessor(self, t):
        f = self.frame_of(t)
        if self.tr.ball_seen[f]:
            d = np.linalg.norm(self.tr.players[f] - self.tr.ball[f], axis=1)
            d = np.where(np.isnan(d), 99, d)
            i = int(np.argmin(d))
            if d[i] < 3.0:
                self.carrier = (i, t)
                return i
        if self.carrier and t - self.carrier[1] < 0.8:
            return self.carrier[0]
        return None

    def ledger_rows(self, t):
        cur = None
        for tt, l in self.timeline:
            if tt <= t:
                cur = l
        if cur is None:
            return None
        rows = []
        for pid, a in cur["rows"].items():
            col = "#F5B63A" if a["role"] == "GK" else "#4FD6A0"
            b = cur["before"][pid]
            d = a["decomposition"]
            dl = a["pulse"]["mean"] - b["pulse"]["mean"]
            rows.append(dict(name=a["name_ar"], obs=ltr(f"{a['pulse_observed']:+.2f}"),
                             sub=f"الحظ يفسر {ltr(str(round(d['luck_share']['mean'] * 100)) + '%')} · أثر التسديدة {ltr(format(dl, '+.2f'))}",
                             lo=a["pulse"]["lo"], hi=a["pulse"]["hi"], obsv=a["pulse_observed"], gate=d["gate"], color=col))
        return rows

    def state(self, t):
        gl = [(e.t + 1.2, e.team) for e in self.shots if e.detail.get("outcome") == "goal"]
        sc = {"KBY": sum(1 for g, tm in gl if g <= t and tm == "KBY"), "RMR": sum(1 for g, tm in gl if g <= t and tm == "RMR")}
        i = self.possessor(t)
        f = self.frame_of(t)
        if i is None:
            rows = [["السرعة", "—", ""], ["المسافة", "—", ""], ["أقرب ضاغط", "—", ""]]
            title = "قراءات اللاعب"
        else:
            pid = self.tr.ids[i]
            team = 0 if pid.startswith("kby") else 1
            opp = [j for j in range(16) if (self.tr.ids[j].startswith("kby")) != (team == 0)]
            dd = np.linalg.norm(self.tr.players[f][opp] - self.tr.players[f][i], axis=1)
            dd = dd[~np.isnan(dd)]
            title = self.names[pid]["name_ar"]
            rows = [["السرعة", ltr(f"{self.speed[f, i] * 3.6:.0f}"), "كم/س"],
                    ["المسافة", ltr(f"{self.cum[f, i]:.0f}"), "م"],
                    ["أقرب ضاغط", ltr(f"{dd.min():.1f}" if len(dd) else "—"), "م"]]
        shot = None
        for k, e in enumerate(self.shots):
            if e.t + 0.8 <= t <= e.t + 4.3:
                lab, colr = OUT.get(e.detail.get("outcome"), ("تسديدة", "#A9C4B3"))
                shot = dict(title=f"تسديدة {self.names[e.player]['name_ar']}", outcome=lab, color=colr,
                            detail=f"{ltr('xG ' + format(self.xg.get(k, 0), '.2f'))} · {ltr(format(e.detail['speed_ms'] * 3.6, '.0f'))} كم/س · {ltr(format(e.detail['distance'], '.0f'))} م")
        cap = None
        for u in self.utt:
            s0 = u["start"] - T0
            if s0 <= t <= u["end"] - T0 + 0.4:
                cap = dict(kind="goal" if u["kind"] == "goal" else u["kind"], cut=bool(u["cut_off"]),
                           who="تعليق طارق" if u["kind"] != "ledger" else "دفتر المهارة والحظ",
                           min=clock(t), ar=u["text"], en=u["en"])
        lanes = []
        for prof, lab in (("casual", "مشجّع"), ("blind", "وصف صوتي"), ("investor", "مستثمر")):
            bars = []
            for u in window(self.ar, prof, self.dur):
                x0, x1 = (u["start"] - T0) / self.dur, (u["end"] - T0) / self.dur
                if x0 <= t / self.dur:
                    bars.append(dict(x0=x0, x1=min(x1, t / self.dur), led=u["kind"] == "ledger", cut=bool(u["cut_off"])))
            lanes.append(dict(label=lab, bars=bars, drops=[]))
        return dict(teams=self.teams, score=sc, clock=clock(t), tag={"investor": "مستثمر", "casual": "مشجّع عادي", "blind": "وصف صوتي"}[self.profile],
                    metrics=rows, t=dict(metrics=title, radar="الرادار", ledger="دفتر المهارة والحظ", lanes="خط النَّفَس"),
                    shot=shot, ledger=self.ledger_rows(t), cap=cap, lanes=lanes, goals=[g / self.dur for g in self.goals if g <= t])


class Overlay:
    def __init__(self, tr, Hm, evs):
        self.tr, self.evs = tr, evs
        self.Hi = np.linalg.inv(Hm)
        n = len(tr.ball)
        P = tr.players.reshape(-1, 2)
        ok = ~np.isnan(P[:, 0])
        img = np.full((len(P), 2), np.nan, np.float32)
        img[ok] = cv2.perspectiveTransform(P[ok].astype(np.float32).reshape(-1, 1, 2), self.Hi).reshape(-1, 2) * SC
        self.pimg = img.reshape(n, 16, 2)
        b = tr.ball.astype(np.float32)
        okb = ~np.isnan(b[:, 0])
        self.bimg = np.full((n, 2), np.nan, np.float32)
        self.bimg[okb] = cv2.perspectiveTransform(b[okb].reshape(-1, 1, 2), self.Hi).reshape(-1, 2) * SC
        self.team = [0 if i.startswith("kby") else 1 for i in tr.ids]
        self.radar_bg = self._radar_bg()

    def to_img(self, pts):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), self.Hi).reshape(-1, 2) * SC

    def _radar_bg(self):
        x, y, w, h = RADAR
        im = np.zeros((h, w, 3), np.uint8)
        im[:] = (52, 92, 24)
        s = w / PITCH_L
        L = (220, 235, 225)
        P = lambda a, b: (int(a * s), int(b * s))
        cv2.rectangle(im, P(0, 0), P(PITCH_L - 1e-3, PITCH_W - 1e-3), L, 1, cv2.LINE_AA)
        cv2.line(im, P(PITCH_L / 2, 0), P(PITCH_L / 2, PITCH_W), L, 1, cv2.LINE_AA)
        cv2.circle(im, P(PITCH_L / 2, PITCH_W / 2), int(9.15 * s), L, 1, cv2.LINE_AA)
        for gx, sgn in ((0, 1), (PITCH_L, -1)):
            cv2.rectangle(im, P(gx, PITCH_W / 2 - 20.16), P(gx + sgn * 16.5, PITCH_W / 2 + 20.16), L, 1, cv2.LINE_AA)
            cv2.rectangle(im, P(gx, PITCH_W / 2 - 9.16), P(gx + sgn * 5.5, PITCH_W / 2 + 9.16), L, 1, cv2.LINE_AA)
        return im

    def draw(self, frame, f, t, poss, hud_rgba_cache):
        tr = self.tr
        ov = frame.copy()
        # shot cone (first 2.4 s after each shot): shooter -> goal mouth
        for e in self.evs:
            if e.kind == "shot" and 0 <= t - e.t <= 2.4:
                gx = PITCH_L if e.detail["x"] > PITCH_L / 2 else 0.0
                i = tr.ids.index(e.player)
                a = tr.players[min(int(e.t * tr.fps), len(tr.ball) - 1), i]
                if np.isnan(a[0]):
                    continue
                poly = self.to_img([a, [gx, PITCH_W / 2 - GOAL_W / 2], [gx, PITCH_W / 2 + GOAL_W / 2]]).astype(np.int32)
                col = {"goal": GOLD, "save": MINT}.get(e.detail.get("outcome"), FLARE)
                cv2.fillConvexPoly(ov, poly, col, cv2.LINE_AA)
        # pass in flight
        for e in self.evs:
            if e.kind == "pass" and e.t <= t <= e.detail["t_received"] + 0.5:
                i, j = tr.ids.index(e.player), tr.ids.index(e.detail["to"])
                fi = min(int(e.t * tr.fps), len(tr.ball) - 1)
                fj = min(int(e.detail["t_received"] * tr.fps), len(tr.ball) - 1)
                a, b = self.pimg[fi, i], self.pimg[fj, j]
                if not (np.isnan(a[0]) or np.isnan(b[0])):
                    cv2.line(ov, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(ov, 0.30, frame, 0.70, 0)
        for i in range(16):
            p = self.pimg[f, i]
            if np.isnan(p[0]):
                continue
            c = TEAM_BGR[self.team[i]]
            sx = 15 + 20 * max(0.0, (p[1] / OH) - 0.15)
            cv2.ellipse(frame, (int(p[0]), int(p[1])), (int(sx), int(sx * 0.42)), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.ellipse(frame, (int(p[0]), int(p[1])), (int(sx - 3), int(sx * 0.42 - 2)), 0, 0, 360, c, -1 if i == poss else 2, cv2.LINE_AA)
        if poss is not None:
            p = self.pimg[f, poss]
            if not np.isnan(p[0]):
                x, y = int(p[0]), int(p[1] - 66)
                cv2.fillConvexPoly(frame, np.array([[x - 11, y - 12], [x + 11, y - 12], [x, y + 6]], np.int32), (255, 255, 255), cv2.LINE_AA)
        # ball trail
        pts = [self.bimg[g] for g in range(max(0, f - 14), f + 1) if not np.isnan(self.bimg[g][0])]
        for k in range(1, len(pts)):
            cv2.line(frame, tuple(pts[k - 1].astype(int)), tuple(pts[k].astype(int)), (255, 255, 255), max(1, k // 3), cv2.LINE_AA)
        if pts:
            cv2.circle(frame, tuple(pts[-1].astype(int)), 11, (255, 255, 255), 2, cv2.LINE_AA)
        # radar dots
        x, y, w, h = RADAR
        r = self.radar_bg.copy()
        s = w / PITCH_L
        for i in range(16):
            q = tr.players[f, i]
            if not np.isnan(q[0]):
                cv2.circle(r, (int(q[0] * s), int(q[1] * s)), 5 if i != poss else 7, TEAM_BGR[self.team[i]], -1, cv2.LINE_AA)
                if i == poss:
                    cv2.circle(r, (int(q[0] * s), int(q[1] * s)), 9, (255, 255, 255), 1, cv2.LINE_AA)
        if tr.ball_seen[f]:
            cv2.circle(r, (int(tr.ball[f, 0] * s), int(tr.ball[f, 1] * s)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return frame, r

    def paste_radar(self, frame, r):
        x, y, w, h = RADAR
        frame[y:y + h, x:x + w] = r


def render(video, out, cache=None, audio=True, profile="investor", voice=None, cuts=(), progress=print):
    tr, evs, Hm = pipeline.analyse(video, cache=cache)
    facts, ar, en = build_sessions(evs)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / tr.fps
    story = Story(tr, evs, facts, ar, en, dur, profile)
    if voice:                                    # captions stay up while the recorded line is spoken; the last frame holds if needed
        for u, s in zip(story.utt, recorded_segments(voice, cuts, 24000)):
            u["end"] = u["start"] + len(s) / 24000 + 0.15
        dur = max(dur, max(u["end"] for u in story.utt) - T0 + 1.0)
        story.dur = dur
    total = int(dur * tr.fps)
    ov = Overlay(tr, Hm, evs)
    hud = Hud()
    cache_h: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    silent = str(Path(out).with_suffix(".silent.mp4"))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                           "-r", str(tr.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", silent],
                          stdin=subprocess.PIPE)
    f = 0
    last = None
    state_key = ""
    n_states = 0
    while True:
        if f >= total:
            break
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        fd = min(f, len(tr.ball) - 1)
        t = f / tr.fps
        fr = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_CUBIC)
        if f % 6 == 0 or not state_key:
            st = story.state(t)
            state_key = json.dumps(st, sort_keys=True, ensure_ascii=False)
            if state_key not in cache_h:
                cache_h.clear()
                n_states += 1
                rgba = hud.render(state_js(st))
                a = rgba[..., 3:4].astype(np.float32) / 255
                cache_h[state_key] = (1 - a, rgba[..., :3][..., ::-1].astype(np.float32) * a)
        poss = story.possessor(t)
        fr, radar = ov.draw(fr, fd, t, poss, cache_h)
        inv_a, premul = cache_h[state_key]
        fr = (fr.astype(np.float32) * inv_a + premul).astype(np.uint8)
        ov.paste_radar(fr, radar)
        px = int(LANE_X0 + (t / dur) * LANE_W)
        cv2.line(fr, (px, 903), (px, 1045), (255, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(fr.tobytes())
        f += 1
        if f % 100 == 0:
            progress(f"{f}/{total} frames, {n_states} HUD states")
    ff.stdin.close()
    ff.wait()
    hud.close()
    if audio:
        wav = narration(story.utt, dur, voice=voice, cuts=cuts)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                        "-shortest", out], check=True)
        Path(silent).unlink()
    else:
        Path(silent).rename(out)
    return dict(frames=f, seconds=round(dur, 1), hud_states=n_states, utterances=len(story.utt))


def recorded_segments(path, cuts, rate):
    """Cut one continuous recording of the commentary lines into per-line clips (cuts = seconds between lines)."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5", "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    edges = [0.0, *cuts, len(x) / rate]
    out = []
    for a, b in zip(edges, edges[1:]):
        s = x[int(a * rate):int(b * rate)]
        loud = np.where(np.abs(s) > 0.02)[0]
        if len(loud):
            s = s[max(0, loud[0] - int(0.06 * rate)):loud[-1] + int(0.1 * rate)]
        out.append(s)
    return out


def narration(utts, dur, rate=24000, voice=None, cuts=()):
    """Piper clips placed at the moment each caption starts, plus a soft crowd bed."""
    pp = None
    segs = recorded_segments(voice, cuts, rate) if voice else None
    if segs is None:
        from scripts.build_audio import Piper
        pp = Piper()
    buf = np.zeros(int((dur + 1) * rate), np.float32)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, len(buf)).astype(np.float32)
    k = np.ones(40, np.float32) / 40
    bed = np.convolve(noise, k, mode="same") * 0.035
    buf += bed
    for n, u in enumerate(utts):
        if segs is not None:
            if n >= len(segs):
                continue
            a = segs[n]
        else:
            pcm, r = pp.wav(u["text"], "ar")
            a = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
            if r != rate:
                a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * rate / r)), np.arange(len(a)), a)
        i0 = int((u["start"] - T0) * rate)
        seg = slice(i0, i0 + len(a))
        buf[seg] *= 0.45                                    # duck the crowd under the voice
        buf[i0:i0 + len(a)] += a[: len(buf) - i0] * 0.95
    p = str(Path("/tmp") / "nafas_narration.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(buf, -1, 1) * 32767).astype("<i2").tobytes())
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--cache")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--profile", default="investor")
    ap.add_argument("--voice", help="recording of the caption lines, read in order")
    ap.add_argument("--cuts", default="", help="comma-separated seconds where one line ends and the next begins")
    a = ap.parse_args()
    print(render(a.video, a.out, a.cache, not a.no_audio, a.profile, a.voice, tuple(float(c) for c in a.cuts.split(",") if c)))
