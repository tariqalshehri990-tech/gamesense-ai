"""Narrate the six submission-video lines in YOUR cloned voice with a local VoiceStudio backend.

VoiceStudio (https://github.com/debpalash/VoiceStudio, AGPL-3.0) runs on your own machine, so the recording never leaves it.
Start the backend first (desktop app, or Docker: see docs/install/docker.md in that repo), then:

    python scripts/voicestudio_voiceover.py --ref my_voice.m4a [--url http://127.0.0.1:3900] [--out narration.mp3]

Set OMNIVOICE_API_KEY if the backend is protected. Only clone a voice you own: the script asks you to confirm.
Writes line1.wav..line6.wav and narration.mp3 (six lines, 2 s apart) for `scripts/build_submission_video.py --voice`.
"""
import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lines() -> list[str]:
    rows = [l for l in (ROOT / "DEMO_SCRIPT.md").read_text(encoding="utf-8").splitlines() if re.match(r"\| [1-6] \|", l)]
    return [r.split("|")[4].strip() for r in rows]


def call(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 900) -> bytes:
    h = dict(headers or {})
    if os.environ.get("OMNIVOICE_API_KEY"):
        h["Authorization"] = "Bearer " + os.environ["OMNIVOICE_API_KEY"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, h), timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{url} -> HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")


def multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    out += (f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    return out, f"multipart/form-data; boundary={b}"


def prepare_reference(src: Path, seconds: float, start: float) -> Path:
    """Mono 24 kHz WAV of a clean stretch of the recording (a short, clean clip clones best)."""
    out = Path("reference.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds), "-i", str(src),
                    "-ac", "1", "-ar", "24000", str(out)], check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="your own voice recording (any ffmpeg-readable file)")
    ap.add_argument("--url", default=os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900"))
    ap.add_argument("--start", type=float, default=2.0, help="where the clean speech starts in the recording (s)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--ref-text", default="", help="what is said in the reference clip (auto-transcribed if empty)")
    ap.add_argument("--name", default="Tariq")
    ap.add_argument("--out", default="narration.mp3")
    ap.add_argument("--yes", action="store_true", help="confirm the recording is your own voice")
    a = ap.parse_args()
    if not a.yes and input("Is this recording your own voice, and do you allow cloning it? [y/N] ").strip().lower() != "y":
        raise SystemExit("cancelled")
    base = a.url.rstrip("/")
    health = json.loads(call(base + "/health", timeout=15))
    print("backend:", health)

    ref = prepare_reference(Path(a.ref), a.seconds, a.start)
    body, ctype = multipart({"name": a.name, "kind": "clone", "language": "Auto", "ref_text": a.ref_text}, "ref_audio", ref)
    prof = json.loads(call(base + "/profiles", body, {"Content-Type": ctype}))
    pid = prof.get("id") or prof.get("profile_id")
    assert pid, prof
    print("voice profile:", pid)

    files = []
    for i, text in enumerate(lines(), 1):
        req = json.dumps({"model": "tts-1", "voice": pid, "input": text, "response_format": "wav"}).encode()
        f = Path(f"line{i}.wav")
        f.write_bytes(call(base + "/v1/audio/speech", req, {"Content-Type": "application/json"}))
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                             capture_output=True, text=True, check=True).stdout.strip()
        print("wrote", f, dur, "s")
        files.append(f)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "2", "-i", "anullsrc=r=24000:cl=mono", "gap.wav"], check=True)
    Path("list.txt").write_text("".join(f"file '{f}'\n" + ("file 'gap.wav'\n" if i < 5 else "") for i, f in enumerate(files)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt", "-ar", "24000",
                    "-c:a", "libmp3lame", a.out], check=True)
    print("wrote", a.out, "- listen before using it")


if __name__ == "__main__":
    main()
