import streamlit as st

st.set_page_config(page_title="GameSense AI", page_icon="🎮")

st.title("🎮 GameSense AI")
st.subheader("ذكاء اصطناعي يحلل سلوك اللاعب ويساعدك تفوز")

st.markdown("---")

game = st.selectbox("اختر لعبتك:", ["PUBG", "Valorant", "Fortnite", "League of Legends"])

skill = st.slider("مستوى مهارتك من 1-10:", 1, 10, 5)

style = st.radio("أسلوب لعبك:", ["هجومي", "دفاعي", "متوازن"])

if st.button("حلّل أدائي 🔍"):
    st.success(f"تم التحليل!")
    st.write(f"🎯 اللعبة: {game}")
    st.write(f"📊 مستواك: {skill}/10")
    st.write(f"🧠 أسلوبك: {style}")
    st.info("💡 نصيحة ذكية: ركز على الـ Aim training 15 دقيقة يومياً، وبتشوف فرق كبير في أسبوع!")
    st.balloons()

st.markdown("---")
st.caption("تم التطوير لـ AI Mini-Project Competition")