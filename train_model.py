import pandas as pd
import joblib
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("Career Guidance Dataset.csv")

# Preprocess
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["text"] = (df["question"] + " " + df["answer"]).apply(preprocess)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["role"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "intent_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
