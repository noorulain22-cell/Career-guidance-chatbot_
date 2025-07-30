
import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Load trained model and vectorizer
model = joblib.load("career_chatbot_project/intent_model.pkl")
vectorizer = joblib.load("career_chatbot_project/vectorizer.pkl")

# Streamlit interface
st.title("Career Guidance Chatbot")
st.write("Ask a question to get a suggested career role.")

# Input field
user_input = st.text_input("Enter your question:")

if user_input:
    clean_input = preprocess(user_input)
    vec_input = vectorizer.transform([clean_input])
    prediction = model.predict(vec_input)[0]
    st.success(f"Suggested Career Role: {prediction}")
