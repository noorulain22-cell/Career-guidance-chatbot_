import streamlit as st
import joblib
import string

# Load model and vectorizer
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

st.title("Career Guidance Chatbot")
user_input = st.text_input("Ask me something about your career:")

if user_input:
    clean_text = preprocess(user_input)
    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    st.write(f"🔍 Suggested Role: **{prediction}**")
