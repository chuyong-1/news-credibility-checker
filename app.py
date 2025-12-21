import streamlit as st
import joblib

st.set_page_config(
    page_title="AI News Credibility Checker",
    page_icon="📰",
    layout="centered"
)

st.title("📰 AI News Credibility Checker")
st.write("Paste a news article below to check if it is **Fake or Real**.")

# Load model (NO text cleaning yet)
@st.cache_resource
def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

news = st.text_area("News Text")

if st.button("Check Credibility"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([news])
        prediction = model.predict(vec)

        if prediction[0] == 1:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
