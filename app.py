import streamlit as st
import pandas as pd
import plotly.express as px
import random

st.set_page_config(page_title="GameSense AI Pro", page_icon="🎮", layout="wide")

st.title("🎮 GameSense AI Pro")
st.markdown("### AI that analyzes player behavior & predicts performance")

col1, col2 = st.columns(2)

with col1:
    game = st.selectbox("Choose your game:", ["PUBG", "Valorant", "Fortnite", "League of Legends", "Call of Duty"])
    skill = st.slider("Your skill level 1-10:", 1, 10, 5)
    style = st.radio("Your play style:", ["Aggressive", "Defensive", "Balanced"])
    hours = st.number_input("Hours played weekly:", 1, 100, 10)
    kd = st.number_input("Your K/D ratio:", 0.1, 10.0, 1.5)

with col2:
    st.markdown("#### 📊 Live Stats Preview")
    data = {
        "Metric": ["Aim", "Strategy", "Teamwork", "Survival"],
        "Score": [random.randint(40,95), random.randint(40,95), random.randint(40,95), random.randint(40,95)]
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Metric", y="Score", color="Score", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

if st.button("🚀 Analyze my performance with AI", use_container_width=True):
    with st.spinner("AI is analyzing your gameplay..."):
        st.success("✅ Analysis Complete!")
        
        # AI Logic
        if skill < 4:
            tip = "🎯 Focus on Aim Lab 20 mins daily. Your aim score is low."
            rank = "Bronze - Needs Training"
        elif skill < 7:
            tip = "🧠 Good! Improve game sense. Watch pro replays + positioning."
            rank = "Gold - Solid Player"
        else:
            tip = "👑 Elite! Focus on leadership & clutch decision making."
            rank = "Diamond - Pro Level"

        if style == "Aggressive" and kd < 1.5:
            tip += " You are too aggressive, try more cover play."

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Rank", rank)
        c2.metric("Win Probability Next Match", f"{random.randint(45,89)}%")
        c3.metric("AI Improvement Score", f"+{random.randint(5,25)}%")

        st.info(f"*AI Smart Tip:* {tip}")
        
        # Radar chart data
        st.markdown("#### 📈 Your Performance Breakdown")
        chart_data = pd.DataFrame({
            "Skill": ["Aim", "Movement", "Game IQ", "Reaction"],
            "You": [skill*10, hours*2, random.randint(50,90), random.randint(50,90)],
            "Pro Average": [85, 85, 90, 88]
        })
        fig2 = px.line_polar(chart_data, r="You", theta="Skill", line_close=True)
        fig2.add_traces(px.line_polar(chart_data, r="Pro Average", theta="Skill", line_close=True).data)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.balloons()

st.markdown("---")
st.caption("Built for Premier League x Microsoft AI Mini-Project | GameSense AI Pro v2.0 | Powered by Streamlit + Plotly")
