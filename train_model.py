import pandas as pd
import joblib
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")
from nltk.corpus import stopwords

# Load datasets
fake = pd.read_csv("data/Fake.csv", nrows=2000)
true = pd.read_csv("data/True.csv", nrows=2000)

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)

df = df.dropna(subset=["title", "text"])

df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Cleaning text...")
df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

print("Label distribution:")
print(y.value_counts())

vectorizer = TfidfVectorizer(max_features=2000, min_df=10)
X_vec = vectorizer.fit_transform(X)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model trained and saved successfully")
