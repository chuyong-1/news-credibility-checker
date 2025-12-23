import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

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
    - Uses a machine learning model trained on real and fake news  
    - Applies NLP techniques (TF-IDF + Logistic Regression)  
    - Returns probability-based credibility scores  
    - Explains *why* the prediction was made using Explainable AI (LIME)

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
# LIME Explainer
# ----------------------------------
class_names = ["Real", "Fake"]
explainer = LimeTextExplainer(class_names=class_names)

def predict_proba_text(texts):
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

# ----------------------------------
# User Input
# ----------------------------------
news = st.text_area(
    "Paste the full news article here 👇",
    height=250,
    placeholder="Enter news content to analyze credibility..."
)

# ----------------------------------
# Prediction + Explainability
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
        # Main Result
        # ----------------------------------
        st.subheader(f"📊 Credibility Score: {credibility_score}%")
        st.progress(credibility_score / 100)

        if credibility_score >= 75:
            st.success(f"✅ Likely Real News")
        elif credibility_score >= 45:
            st.warning(f"⚠️ Uncertain Credibility")
        else:
            st.error(f"❌ Likely Fake News")

        # ----------------------------------
        # Percentage Breakdown
        # ----------------------------------
        st.subheader("📈 Credibility Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🟢 Real News Probability",
                value=f"{real_prob:.2f}%"
            )

        with col2:
            st.metric(
                label="🔴 Fake News Probability",
                value=f"{fake_prob:.2f}%"
            )

        # ----------------------------------
        # Explainability (LIME)
        # ----------------------------------
        st.subheader("🧠 Why did the model decide this?")

        with st.spinner("Generating explanation..."):
            explanation = explainer.explain_instance(
                news,
                predict_proba_text,
                num_features=10
            )

        exp_list = explanation.as_list()

        st.markdown("**Top influencing words:**")
        for word, weight in exp_list:
            if weight > 0:
                st.markdown(f"🟢 **{word}** → pushes towards *Real*")
            else:
                st.markdown(f"🔴 **{word}** → pushes towards *Fake*")

        st.caption(
            "LIME highlights influential words based on local approximations of the model."
        )
