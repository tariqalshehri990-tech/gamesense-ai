"""Azure AI Speech text-to-speech (REST). Enabled only when AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are set.

Saudi-locale Azure voices read Modern Standard Arabic, not Saudi dialect. The text is Saudi-flavoured; the audio is not.
"""
from __future__ import annotations

import os
import urllib.request
from xml.sax.saxutils import escape

VOICES = {"ar": ("ar-SA", "ar-SA-HamedNeural"), "en": ("en-GB", "en-GB-RyanNeural")}
MAX_CHARS = 400


def enabled() -> bool:
    return bool(os.environ.get("AZURE_SPEECH_KEY") and os.environ.get("AZURE_SPEECH_REGION"))


def build_ssml(text: str, lang: str, rate: str = "+8%") -> str:
    locale, voice = VOICES[lang]
    return (f"<speak version='1.0' xml:lang='{locale}'><voice name='{voice}'>"
            f"<prosody rate='{rate}'>{escape(text[:MAX_CHARS])}</prosody></voice></speak>")


def synthesize(text: str, lang: str) -> bytes:
    region, key = os.environ["AZURE_SPEECH_REGION"], os.environ["AZURE_SPEECH_KEY"]
    req = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=build_ssml(text, lang).encode("utf-8"),
        headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3", "User-Agent": "nafas"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()
