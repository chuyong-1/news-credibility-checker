import streamlit as st
import joblib
import numpy as np

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AI News Credibility Checker",
    page_icon="📰",
    layout="centered"
)

# ----------------------------------
# Title & Description
# ----------------------------------
st.title("📰 AI News Credibility Checker")

st.markdown(
    """
    **How it works**
    - Uses a machine learning model trained on real and fake news articles  
    - Analyzes language patterns using NLP  
    - Returns a probability-based credibility score  

    ⚠️ *This tool assists credibility assessment and does not replace human fact-checking.*
    """
)

# ----------------------------------
# Load Model & Vectorizer
# ----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------------
# User Input
# ----------------------------------
news = st.text_area(
    "Paste the full news article here 👇",
    height=250,
    placeholder="Enter news content to analyze credibility..."
)

# ----------------------------------
# Prediction Logic
# ----------------------------------
if st.button("🔍 Check Credibility"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vec = vectorizer.transform([news])

        # Probability prediction
        proba = model.predict_proba(vec)[0]
        real_prob = proba[0] * 100
        fake_prob = proba[1] * 100

        credibility_score = round(real_prob, 2)

        # ----------------------------------
        # Results
        # ----------------------------------
        st.subheader("📊 Credibility Analysis")
        st.progress(credibility_score / 100)

        if credibility_score >= 75:
            st.success(f"✅ Likely Real News ({credibility_score}%)")
        elif credibility_score >= 45:
            st.warning(f"⚠️ Uncertain Credibility ({credibility_score}%)")
        else:
            st.error(f"❌ Likely Fake News ({credibility_score}%)")

        # Extra details
        with st.expander("📌 See detailed probabilities"):
            st.write(f"**Real News Probability:** {round(real_prob, 2)}%")
            st.write(f"**Fake News Probability:** {round(fake_prob, 2)}%")

        st.caption(
            "⚠️ This prediction is AI-generated and should be used as a supporting signal only."
        )
