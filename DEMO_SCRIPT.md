# Submission video (target about 1:45, hard limit 2:00)

Built by `scripts/build_submission_video.py` from `scripts/video_cards.py` (cards), live recordings of the running app, and `demo/nafas_vision.mp4`. Narration is English; the Arabic commentary in scene 3 carries English subtitles.

| # | Scene | On screen | Narration |
|---|---|---|---|
| 1 | Problem | Title, then three problems | Football commentary talks about everything, all the time. Viewers get lost. Blind fans miss the goal. Clubs pay for luck. This is Tariq, named for the morning star that cuts through the dark. |
| 2 | How it works | Diagram builds in three steps | Tariq treats attention as a budget. We generate a synthetic match, render it as broadcast video, and track every player and the ball with computer vision. Events flow into a team of agents, coordinated by a conductor, and every fact is served over MCP. |
| 3 | Vision | Tracked synthetic broadcast, rings, radar, ledger card, Arabic commentary with English subtitles | Here the tracker reads a synthetic broadcast. It finds the shot, the save and the goal, and scores every chance with expected goals. Tariq then comments in Arabic, with English subtitles. |
| 4 | Live app | Casual-fan tab, breath line with cut-off marks | The same match, live in the app. When a bigger moment arrives, Tariq cuts its own sentence, shortens a stale line, or stays silent. The breath line shows every choice. |
| 5 | Investor | Ledger, "What if?" | For investors, the same events feed a skill versus luck ledger. Each player gets a range, not a verdict. With too few shots, Tariq says it does not know. Press What if on a goal, and see how much was luck. |
| 6 | Agents and Azure | Agent trace, then "Built with" card | Under the hood: a scout, an explainer, an editor, a dialect desk and an auditor, with every handoff logged. Facts are served over MCP. Phrasing can optionally run on Microsoft Foundry through Agent Framework, the app ships as a container for Azure Container Apps, and Azure AI Speech gives it a voice. All data is synthetic. The code is public on GitHub. |

## Official rules checklist (OFFICIAL RULES.md, microsoft/insidethegamehackathon)
- Under two minutes, public URL on YouTube, Vimeo, Facebook Video or Youku (section 4.4/4.9): upload and paste the link on the project page.
- Shows the project functioning (section 4.4): scenes 3 to 5 are the running app and tracker.
- English materials (section 4.9): English narration, English UI in recordings, English subtitles on Arabic speech.
- Synthetic, football-realistic data only (section 4.7): "Synthetic data" badge on every live scene.
- No third-party trademarks or copyrighted music or footage (section 4.4): none used; narration is the entrant's own voice.
- Public GitHub repository (section 4.4/4.8) and a working project for judges to test.
- Pitch names the Microsoft/Azure technologies, what it does, and the problem it solves.
