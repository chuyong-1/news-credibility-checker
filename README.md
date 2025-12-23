# 📰 AI News Credibility Checker

An **AI-powered web application** that evaluates the credibility of news articles using **Natural Language Processing (NLP)** and **Explainable AI**.  
The system not only predicts whether news is *Real or Fake*, but also **explains why** the prediction was made.

---

## 🚀 Features

- ✅ Fake vs Real News Classification  
- 📊 Probability-based Credibility Score  
- 📈 Real vs Fake Percentage Breakdown  
- 🧠 Explainable AI using **LIME** (highlights influential words)  
- 🌐 Interactive Web App built with **Streamlit**  
- ♻️ Reproducible Machine Learning pipeline  
- 🧪 Model evaluated using standard ML metrics  

---

## 🧠 How It Works

1. User pastes a news article into the web app  
2. Text is cleaned and preprocessed  
3. Features are extracted using **TF-IDF Vectorization**  
4. A **Logistic Regression** model predicts credibility  
5. Output includes:
   - Credibility Score (%)
   - Real vs Fake probabilities  
6. **LIME** explains which words influenced the decision  

⚠️ This tool provides AI-assisted credibility analysis and does not replace human fact-checking.

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- NLTK  
- Streamlit  
- LIME (Explainable AI)  
- Joblib  
- Git & GitHub  

---

## 📂 Project Structure
news-credibility-checker/
│
├── app.py # Streamlit web application
├── train_model.py # Model training & evaluation
├── requirements.txt # Dependencies
├── README.md # Documentation
├── .gitignore
│
├── data/ # Dataset files
├── model/ # Saved models (ignored in git)
├── screenshots/ # App screenshots


---

## 📊 Model Evaluation

The model is trained using a **train-test split** and evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

This ensures the model generalizes well to unseen data.

---

## 🧠 Explainable AI (Why This Matters)

Instead of acting as a black box, the model explains predictions using **LIME**.

Examples:
- 🟢 *official*, *confirmed*, *report* → pushes towards **Real**
- 🔴 *shocking*, *you won’t believe*, *secret* → pushes towards **Fake**

This improves transparency, trust, and usability.
---
## 🖥️ How to Run Locally
### 1️⃣ Clone the repository
```bash
git clone https://github.com/chuyong-1/news-credibility-checker.git
cd news-credibility-checker

2️⃣ Create and activate a virtual environment
python -m venv venv
Windows
venv\Scripts\activate
Mac / Linux
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run app.py

## 📸 Screenshots

### ✅ Real News Prediction
![Real News Example](screenshots/real_news_example.png)
### 🧠 Explainable AI (LIME)
![Explainability View](screenshots/t_explainability_view.png)

### ❌ Fake News Prediction
![Fake News Example](screenshots/fake_news_example.png)

### 🧠 Explainable AI (LIME)
![Explainability View](screenshots/f_explainability_view.png)
