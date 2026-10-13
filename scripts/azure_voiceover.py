"""Narrate the six submission-video lines with Azure AI Speech (run where the Azure endpoint is reachable).

    AZURE_SPEECH_KEY=... AZURE_SPEECH_REGION=... python scripts/azure_voiceover.py [--voice en-US-AndrewNeural] [--out narration.mp3]

Writes line1.mp3..line6.mp3 and, if ffmpeg is installed, narration.mp3 (the six lines with two seconds of silence between),
which `scripts/build_submission_video.py --voice narration.mp3` cuts back into scenes.
"""
import argparse
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent


def lines() -> list[str]:
    rows = [l for l in (ROOT / "DEMO_SCRIPT.md").read_text(encoding="utf-8").splitlines() if re.match(r"\| [1-6] \|", l)]
    return [r.split("|")[4].strip() for r in rows]


def tts(text: str, voice: str) -> bytes:
    region, key = os.environ["AZURE_SPEECH_REGION"], os.environ["AZURE_SPEECH_KEY"]
    ssml = f"<speak version='1.0' xml:lang='en-US'><voice name='{voice}'>{escape(text)}</voice></speak>"
    req = urllib.request.Request(f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", ssml.encode("utf-8"),
                                 {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                                  "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3", "User-Agent": "nafas"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--out", default="narration.mp3")
    a = ap.parse_args()
    ls = lines()
    assert len(ls) == 6, ls
    files = []
    for i, t in enumerate(ls, 1):
        f = Path(f"line{i}.mp3")
        f.write_bytes(tts(t, a.voice))
        files.append(f)
        print("wrote", f)
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "2", "-i", "anullsrc=r=24000:cl=mono", "-c:a", "libmp3lame", "gap.mp3"], check=True)
        Path("list.txt").write_text("".join(f"file '{f}'\n" + ("file 'gap.mp3'\n" if i < 5 else "") for i, f in enumerate(files)))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "list.txt", "-ar", "24000", "-c:a", "libmp3lame", a.out], check=True)
        print("wrote", a.out)
    except Exception as e:  # ffmpeg missing: send the six line files instead
        print("could not join lines (", e, "): send line1.mp3..line6.mp3")


if __name__ == "__main__":
    main()
