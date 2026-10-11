# GameSense AI - Autonomous Football Intelligence Studio
### Premier League x Microsoft Hackathon 2026

**Live Demo:** (coming soon)
**Video:** (coming soon)

## The Problem
Fans see *what* happened (2-1). Broadcasters need *why* it happened in <500ms, in Arabic & English.

## The Solution - 3 AI Layers
1. **Live Tactical Narrator (<500ms):** Uses Azure OpenAI in Foundry to detect HIGH_PRESS, OVERLOAD patterns and explains them.
2. **Contextual Story Engine:** Microsoft Fabric queries 5 years PL data to calculate rarity (e.g., "1 in 28 miss") and finds historical parallels.
3. **Predictive Fan Agent:** Cosmos DB predicts Fantasy points and substitution risk.

## Microsoft Stack (Judging Criteria)
- **GitHub + Copilot:** Full development
- **Microsoft Foundry + GPT-4o:** Orchestrator Agent
- **Azure AI Speech:** AR/EN/ES commentary
- **Microsoft Fabric:** Historical data
- **Azure Container Apps:** Deployment

## How to Run
pip install -r requirements.txt
streamlit run app.py