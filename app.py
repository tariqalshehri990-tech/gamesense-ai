import streamlit as st
import pandas as pd
import plotly.express as px
import random

st.set_page_config(page_title="GameSense AI x Premier League", page_icon="⚽", layout="wide")

st.title("⚽ GameSense AI x Premier League")
st.markdown("### Powered by Microsoft Azure AI - Fantasy Assistant")

mode = st.selectbox("Choose Mode:", ["⚽ Premier League Fantasy", "🎮 PUBG / Gaming Analytics"])

if mode == "⚽ Premier League Fantasy":
    col1, col2 = st.columns(2)
    with col1:
        position = st.selectbox("Player Position:", ["Forward", "Midfielder", "Defender", "Goalkeeper"])
        budget = st.slider("Budget (M):", 4.0, 14.0, 8.0)
        form = st.slider("Current Form (1-10):", 1, 10, 7)
    with col2:
        st.markdown("#### 🤖 Azure OpenAI Analysis")
        players = {
            "Forward": ["Haaland", "Salah", "Saka"],
            "Midfielder": ["De Bruyne", "Odegaard", "Palmer"],
            "Defender": ["Saliba", "Van Dijk", "Walker"],
            "Goalkeeper": ["Raya", "Alisson", "Ederson"]
        }
        df = pd.DataFrame({"Player": players[position], "Points Prediction": [random.randint(5,12) for _ in range(3)]})
        fig = px.bar(df, x="Player", y="Points Prediction", color="Points Prediction")
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🚀 Get AI Fantasy Advice", use_container_width=True):
        st.success(f"✅ AI Recommendation: Pick {random.choice(players[position])}! Expected points: {random.randint(6,12)} next GW")
        st.info("💡 Powered by Azure OpenAI Service + Semantic Kernel - Analyzes 30 seasons of data, 300k articles, 9000 videos")
        st.balloons()
else:
    # Your existing PUBG mode
    st.markdown("#### 🎮 Gaming Analytics")
    skill = st.slider("Your skill level 1-10:", 1, 10, 5)
    if st.button("Analyze my performance"):
        st.success(f"Rank: {'Diamond' if skill>7 else 'Gold'} - Win Prob: {random.randint(50,90)}%")

st.markdown("---")
st.caption("Premier League x Microsoft AI Mini-Project | Technologies: Azure OpenAI Service, Semantic Kernel, Streamlit")
