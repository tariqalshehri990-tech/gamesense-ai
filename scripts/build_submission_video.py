"""Assemble the hackathon submission video (<2 min, English narration) from cards, live-app takes and the vision overlay.

    PYTHONPATH=. python scripts/build_submission_video.py --cards DIR --takes DIR --vision demo/nafas_vision.mp4 \
        [--voice narration.m4a] --out demo/nafas_submission.mp4

Without --voice it builds a silent draft with default scene lengths. With --voice (six English lines read in order
with a pause of about two seconds between them) each scene is as long as its line, so picture follows the voice.
"""
import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np

RATE = 24000
DEFAULT = [12.0, 16.0, 15.0, 16.0, 18.0, 19.0]


def sh(*a):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, a)], check=True)


def still_seq(cards, names, d, out):
    per = d / len(names)
    parts = []
    for i, n in enumerate(names):
        p = out.with_name(f"{out.stem}_{i}.mp4")
        sh("-loop", 1, "-t", per, "-i", cards / f"{n}.png", "-vf", "scale=1920:1080,format=yuv420p,fps=25", "-c:v", "libx264", "-crf", 18, p)
        parts.append(p)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    sh("-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", out)


def clip(src, ss, d, out, badge=None, speed=1.0):
    vf = f"scale=1920:1080,fps=25,tpad=stop_mode=clone:stop_duration={d}" + (f",setpts=PTS/{speed}" if speed != 1 else "")
    ins = ["-ss", ss, "-i", src]
    if badge:
        sh(*ins, "-i", badge, "-filter_complex", f"[0:v]{vf}[v];[v][1:v]overlay=W-w-24:H-h-20[o]", "-map", "[o]", "-t", d, "-c:v", "libx264", "-crf", 19, "-pix_fmt", "yuv420p", out)
    else:
        sh(*ins, "-vf", vf, "-t", d, "-c:v", "libx264", "-crf", 19, "-pix_fmt", "yuv420p", out)


def wav_of(path, ss=0, d=None, vol=1.0):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(ss)] + (["-t", str(d)] if d else []) + ["-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, "<i2").astype(np.float32) / 32768
    return x * vol


def pause_cuts(path, n_lines):
    """Cut points = middles of the n_lines-1 longest pauses of the recording."""
    out = subprocess.run(["ffmpeg", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=0.5", "-f", "null", "-"], capture_output=True, text=True).stderr
    st, en = [], []
    for l in out.splitlines():
        if "silence_start" in l:
            st.append(float(l.split("silence_start:")[1]))
        if "silence_end" in l:
            en.append(float(l.split("silence_end:")[1].split("|")[0]))
    gaps = sorted([(e - s, (s + e) / 2) for s, e in zip(st, en) if s > 0.5], reverse=True)[: n_lines - 1]
    return sorted(m for _, m in gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--takes", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--voice")
    ap.add_argument("--cuts", help="comma-separated cut seconds (default: detect from pauses)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cards, takes, out = Path(a.cards), Path(a.takes), Path(a.out)
    work = out.parent / "_work"
    work.mkdir(parents=True, exist_ok=True)
    badge = cards / "badge.png"

    segs = None
    D = DEFAULT
    if a.voice:
        from nafas.vision.compose import recorded_segments
        cuts = [float(c) for c in a.cuts.split(",")] if a.cuts else pause_cuts(a.voice, 6)
        print("cuts", cuts)
        segs = recorded_segments(a.voice, cuts, RATE)
        D = [len(s) / RATE + 0.9 for s in segs]
    print("scene lengths", [round(d, 1) for d in D], "total", round(sum(D), 1))
    assert sum(D) < 119, "over two minutes"

    s = [work / f"s{i}.mp4" for i in range(1, 7)]
    t = D[0]
    still_seq(cards, ["title", "problem1", "problem2", "problem3"], t, s[0])
    still_seq(cards, ["arch1", "arch2", "arch3"], D[1], s[1])
    # scene 3: tracked synthetic broadcast, two excerpts joined
    clip(a.vision, 9.0, min(D[2], 7.0), work / "v_a.mp4", badge)
    clip(a.vision, 36.5, max(D[2] - 7.0, 0.5), work / "v_b.mp4", badge)
    (work / "v.txt").write_text(f"file '{work/'v_a.mp4'}'\nfile '{work/'v_b.mp4'}'\n")
    sh("-f", "concat", "-safe", 0, "-i", work / "v.txt", "-c", "copy", s[2])
    clip(takes / "casual.mp4", 20.0, D[3], s[3], badge)
    clip(takes / "investor.mp4", 11.0, D[4], s[4], badge)
    trace_d = min(7.0, D[5] * 0.4)
    clip(takes / "trace.mp4", 14.0, trace_d, work / "tr.mp4", badge)
    still_seq(cards, ["end"], D[5] - trace_d, work / "end.mp4")
    (work / "t6.txt").write_text(f"file '{work/'tr.mp4'}'\nfile '{work/'end.mp4'}'\n")
    sh("-f", "concat", "-safe", 0, "-i", work / "t6.txt", "-c", "copy", s[5])

    # fade each scene in and out, then join
    faded = []
    for i, p in enumerate(s):
        f = work / f"f{i}.mp4"
        d = D[i]
        sh("-i", p, "-vf", f"fade=t=in:st=0:d=0.25,fade=t=out:st={d-0.25:.2f}:d=0.25", "-t", d, "-c:v", "libx264", "-crf", 19, "-pix_fmt", "yuv420p", f)
        faded.append(f)
    (work / "all.txt").write_text("".join(f"file '{p}'\n" for p in faded))
    silent = work / "all.mp4"
    sh("-f", "concat", "-safe", 0, "-i", work / "all.txt", "-c", "copy", silent)

    # audio: English voice per scene, plus the Arabic commentary under scene 3
    total = sum(D)
    buf = np.zeros(int((total + 1) * RATE), np.float32)
    starts = np.cumsum([0, *D])
    ar = wav_of(a.vision, 9.0, min(D[2], 7.0), 0.30)
    i0 = int(starts[2] * RATE)
    buf[i0:i0 + len(ar)] += ar
    ar2 = wav_of(a.vision, 36.5, max(D[2] - 7.0, 0.5), 0.30)
    j0 = i0 + int(min(D[2], 7.0) * RATE)
    buf[j0:j0 + len(ar2)] += ar2
    if segs:
        for k, sg in enumerate(segs):
            o = int((starts[k] + 0.4) * RATE)
            buf[o:o + len(sg)] += sg[: len(buf) - o] * 0.95
    wp = work / "audio.wav"
    with wave.open(str(wp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(buf, -1, 1) * 32767).astype("<i2").tobytes())
    sh("-i", silent, "-i", wp, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out)
    print("wrote", out)


if __name__ == "__main__":
    main()
