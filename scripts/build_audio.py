"""Pre-render commentary audio for the standalone demo.

Best offline engine: Piper neural voices through sherpa-onnx (`pip install sherpa-onnx`), Arabic ar_JO-kareem-medium and
English en_GB-alan-medium, unpacked under $NAFAS_PIPER_DIR (default /tmp/tts). Models:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models (vits-piper-ar_JO-kareem-medium, vits-piper-en_GB-alan-medium).
Arabic text is vocalised first (scripts/arabic_prep.py) because the Arabic Piper voice expects diacritics.
Fallback engine: espeak-ng (offline, robotic but works on every device; `pip install espeakng-loader`).
Better engine: set AZURE_SPEECH_KEY + AZURE_SPEECH_REGION and pass --azure for Azure AI Speech neural voices.
Writes demo/audio.json = {sha1(lang|text)[:16]: base64 mp3}, which scripts/build_demo.py embeds.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def key(lang: str, text: str) -> str:
    """FNV-1a 32-bit over UTF-16 code units; identical to fnv() in the UI so the page can look clips up offline."""
    h = 0x811C9DC5
    b = text.encode("utf-16-le")
    for i in range(0, len(b), 2):
        h = ((h ^ (b[i] | (b[i + 1] << 8))) * 0x01000193) & 0xFFFFFFFF
    return f"{lang}:{h:08x}"


def all_lines() -> dict[str, tuple[str, str]]:
    from fastapi.testclient import TestClient
    from nafas.attention import PROFILES
    from nafas.server import app
    c = TestClient(app)
    out: dict[str, tuple[str, str]] = {}
    for p in PROFILES:
        for lang in ("ar", "en"):
            r = c.get(f"/api/stream?seed=35&profile={p}&lang={lang}&speed=1000000")
            for l in r.text.split("\n\n"):
                if l.startswith("data: "):
                    it = json.loads(l[6:])
                    if it.get("type") == "utterance":
                        out[key(it.get("lang", lang), it["text"])] = (it.get("lang", lang), it["text"])
    out[key("ar", "الصوت شغّال")] = ("ar", "الصوت شغّال")
    out[key("en", "Voice is on")] = ("en", "Voice is on")
    return out


class Espeak:
    def __init__(self) -> None:
        import espeakng_loader as e
        self.lib = ctypes.CDLL(e.get_library_path())
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.rate = self.lib.espeak_Initialize(2, 0, e.get_data_path().encode(), 0)  # 2 = synchronous, no playback
        self.buf: list[bytes] = []
        CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

        def cb(wav, n, ev):
            if wav and n > 0:
                self.buf.append(ctypes.string_at(wav, n * 2))
            return 0
        self._cb = CB(cb)
        self.lib.espeak_SetSynthCallback(self._cb)

    def wav(self, text: str, lang: str) -> bytes:
        self.lib.espeak_SetVoiceByName(b"ar" if lang == "ar" else b"en-gb")
        self.lib.espeak_SetParameter(1, 215, 0)  # rate (words per minute)
        self.buf = []
        data = text.encode("utf-8") + b"\0"
        self.lib.espeak_Synth(data, len(data), 0, 0, 0, 1, None, None)
        self.lib.espeak_Synchronize()
        return b"".join(self.buf)


class Piper:
    def __init__(self) -> None:
        import sherpa_onnx
        base = Path(os.environ.get("NAFAS_PIPER_DIR", "/tmp/tts"))
        self.tts = {}
        for lang, name in (("ar", "ar_JO-kareem-medium"), ("en", "en_GB-alan-medium")):
            d = base / f"vits-piper-{name}"
            cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / f"{name}.onnx"), tokens=str(d / "tokens.txt"),
                                                           data_dir=str(d / "espeak-ng-data")), num_threads=2))
            self.tts[lang] = sherpa_onnx.OfflineTts(cfg)
        self.unknown: set[str] = set()

    def wav(self, text: str, lang: str) -> tuple[bytes, int]:
        import numpy as np
        if lang == "ar":
            from scripts.arabic_prep import vocalise
            text = vocalise(text, self.unknown)
        a = self.tts[lang].generate(text, sid=0, speed=1.0)
        pcm = (np.clip(np.array(a.samples), -1, 1) * 32767).astype("<i2").tobytes()
        return pcm, a.sample_rate


def to_mp3(pcm: bytes, rate: int) -> bytes:
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "-",
                        "-ac", "1", "-ar", "16000", "-b:a", "40k", "-f", "mp3", "-"], input=pcm, capture_output=True, check=True)
    return p.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--azure", action="store_true")
    ap.add_argument("--espeak", action="store_true", help="use the robotic espeak-ng fallback")
    args = ap.parse_args()
    lines = all_lines()
    clips: dict[str, str] = {}
    if args.azure:
        from nafas import voice
        for k, (lang, text) in lines.items():
            clips[k] = base64.b64encode(voice.synthesize(text, lang)).decode()
    elif args.espeak:
        es = Espeak()
        for k, (lang, text) in lines.items():
            pcm = es.wav(text, lang)
            if pcm:
                clips[k] = base64.b64encode(to_mp3(pcm, es.rate)).decode()
    else:
        pp = Piper()
        for k, (lang, text) in lines.items():
            pcm, rate = pp.wav(text, lang)
            clips[k] = base64.b64encode(to_mp3(pcm, rate)).decode()
        if pp.unknown:
            print("words missing from the Arabic lexicon (spoken without diacritics):", sorted(pp.unknown))
    (ROOT / "demo").mkdir(exist_ok=True)
    (ROOT / "demo/audio.json").write_text(json.dumps(clips))
    print(f"{len(clips)}/{len(lines)} clips, {sum(len(v) for v in clips.values())/1e6:.1f} MB base64")


if __name__ == "__main__":
    main()
