# Nafas (نَفَس) — commentary that budgets attention

Built for the Microsoft × Premier League *Inside the Game* developer hackathon. **Synthetic data only**: fictional clubs (Kestrel Bay, Redmoor Rovers), fictional players and contracts.

## The idea
Most match-commentary projects answer *what should we say?* Nafas answers **what can this viewer still take in?**
Each viewer profile has an attention budget that refills over match time. The Editor decides in match time what to say, when to stay silent, when to **cut a sentence off** because a bigger moment arrived, and when to compress or drop stale items.

Profiles: Casual fan (Arabic), Analyst (English), Audio description (Arabic), Investor (Arabic).

### Investor mode "ميزان"
For each player: a **skill vs luck** split with intervals (Bayesian posterior over a finishing/keeping offset), a contract "pulse" (value delivered ÷ wage, break-even = 1.0), and a **what-if** that flips one shot. Under 100 shots the answer is *no verdict*.

## Pipeline (the 5 required stages)
| Stage | Component |
|---|---|
| Ingest | `simulator.py` synthetic events, `Scout` detects moments |
| Interpret | `xg.py` shot quality, `stats.py` skill/luck engine |
| Explain | `Explainer` calls fact tools, `phrasebook.py` templates, optional Foundry rewrite |
| Render | `attention.py` scheduler → SSE stream → RTL-first UI with a "breath line" |
| Personalize | per-profile budgets, language and style |

Agents: Scout, Explainer, Editor, DialectDesk, Auditor (deterministic: numbers, names and advice wording must derive from facts), Conductor. The fact layer is exposed as an **MCP server** (`python -m nafas.mcp_server --http`).

## Run
```bash
pip install -r requirements.txt
uvicorn nafas.server:app --port 8000      # open http://localhost:8000
pip install -r requirements-dev.txt && pytest
```
Works fully offline (template backend). To enable Foundry phrasing: `pip install agent-framework-foundry azure-identity`, set `FOUNDRY_PROJECT_ENDPOINT` and `FOUNDRY_MODEL`. **That path is untested.**

## Azure
`Dockerfile` included. Deploy: `az containerapp up --name nafas --source . --ingress external --target-port 8000`. Microsoft pieces: Agent Framework + Foundry (optional backend), MCP, Container Apps, Azure AI Speech (voice).

### Voice
The 🔊 button speaks each line. With `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` set, the server's `POST /api/tts` returns Azure AI Speech audio (`ar-SA-HamedNeural`, `en-GB-RyanNeural`). Without them, the browser's own speech voice is used. A cut-off line is cut in the audio too: a new line stops the previous one. The Azure path is **untested against the live service** (no key was available while building); the browser path works where the device has an Arabic voice. Best at 6× or 1× speed.

The standalone `demo/index.html` carries pre-rendered clips so it speaks on any device (some browsers expose no speech voices at all). Clips are made offline with **Piper neural voices** (Arabic `ar_JO-kareem`, English `en_GB-alan`) through `sherpa-onnx`; the Arabic text is vocalised first by `scripts/arabic_prep.py` (a lexicon for our closed phrasebook), because the Arabic voice expects diacritics. Build: `pip install sherpa-onnx`, unpack the two `vits-piper-*` models from the sherpa-onnx `tts-models` release into `/tmp/tts` (or `$NAFAS_PIPER_DIR`), then `python scripts/build_audio.py && python scripts/build_demo.py`. The voice is Jordanian-accented MSA, not Saudi; a Whisper-small round trip recovered about half of the Arabic words exactly (names are the weakest part), so treat it as a demo voice. For neural Saudi-locale quality use Azure: `AZURE_SPEECH_KEY=... AZURE_SPEECH_REGION=... python scripts/build_audio.py --azure`. `--elevenlabs` (needs `ELEVENLABS_API_KEY`) is the other recommended route; neither was run live here. `--espeak` selects the old robotic fallback.

## Validation
- Simulator calibrated to Premier League averages (~25 shots/match, ~0.115 xG/shot, ~11% conversion).
- Skill recovery: simulated players have hidden true skill; 90% intervals covered it ~88–90% of the time; test checks positive-skill players rank above negative.
- 12 automated tests (`pytest`).

## What we refuse to claim
- No price prediction, no buy/sell advice (the Auditor blocks such wording).
- Not real data. The xG model and simulator are related, so recovery is partly circular.
- Goals→money conversion is a labelled synthetic assumption (median positive-value player = break-even).
- The pulse band covers skill uncertainty only. What-if is computed against the full-match ledger.
- Auto-detection of events from video is simulated, not implemented.
- Arabic text is Saudi-flavoured; Azure's ar-SA voices read Modern Standard Arabic, not dialect.

## Video analysis (camera → events → Nafas)

`nafas/vision/` turns match video into the same `Match`/`Shot` facts the rest of Nafas runs on: background subtraction and kit-colour segmentation per frame, pitch-landmark calibration (RANSAC homography), Kalman + Hungarian tracking of 16 players and the ball, then possession / pass / shot / save / goal inference from the ball trajectory (`events.py`), a bridge into the xG + skill-vs-luck engine (`bridge.py`), and `compose.py`, which renders the broadcast overlay (tracking rings, pass lines, shot cones, radar, Arabic/English captions, breath lanes, live ledger card) and mixes the narration.

```
pip install -r requirements-vision.txt && playwright install chromium   # HUD is rendered in Chromium for Arabic shaping
PYTHONPATH=. python -m nafas.vision.compose input.mp4 out.mp4 --cache dets.pkl
```
`demo/nafas_vision.mp4` is the result on a **synthetic** 44 s clip made by `nafas/vision/synth.py` + `render.py` (rule 4.7: synthetic data only; we hold no rights to broadcast footage). Honest limits: the tracker was validated against that generator's own ground truth, which is a circular test; it assumes one static wide camera; real footage (occlusion, camera pan, kit clashes) is untested. Narration is either Piper or a recording of the caption lines: `--voice rec.m4a --cuts 6.14,21.19` (cuts = seconds where one line ends); the shipped video uses a human recording.
