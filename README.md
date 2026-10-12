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

The standalone `demo/index.html` carries pre-rendered clips so it speaks on any device (some browsers expose no speech voices at all). Default clips come from offline **espeak-ng** (robotic, no Saudi dialect; build with `python scripts/build_audio.py`). For neural quality, run `AZURE_SPEECH_KEY=... AZURE_SPEECH_REGION=... python scripts/build_audio.py --azure` and then `python scripts/build_demo.py`.

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
