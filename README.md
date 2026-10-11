# gimesense-ai
GameSense AI - Real-time Premier League intelligence that explains why goals happen. Built with Azure OpenAI, Fabric &amp; AI Speech. &lt;500ms tactical narration in AR/EN/ES.
import streamlit as st
st.set_page_config(page_title="GameSense AI", layout="wide", page_icon="⚽")
st.title("⚽ GameSense AI - Autonomous Football Intelligence Studio")
st.caption("Powered by Azure OpenAI + Fabric + AI Speech | Inside the Game Hackathon")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Match Time", "67:23", "+ Live")
c2.metric("Score", "LIV 2 - 1 ARS")
c3.metric("xG Live", "1.82 vs 0.94")
c4.metric("Pressure", "78%", "High")

t1,t2,t3 = st.tabs(["🎙️ Live Narrator", "📖 Story Engine", "🔮 Fan Agent"])
with t1:
    st.success("Tactical: HIGH PRESS TRIGGER at 66:12 - Robertson presses Odegaard")
    st.info("AR: ليفربول يواصل الضغط العالي! صلاح يتقدم بالكرة")
    st.info("EN: Brilliant high press! Salah steals it, overload ON!")
with t2:
    st.metric("Rarity", "1 in 28 matches")
    st.write("Fabric: After such miss, LIV scores in 10 mins in 62% cases")
with t3:
    st.json({"LIV Win Prob": "74%", "Fantasy": "Salah +3 pts"})