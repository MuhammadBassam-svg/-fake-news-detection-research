import streamlit as st
import pandas as pd
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Title
st.title("📰 Fake News Detector")

# Load pre-trained model and vectorizer
model = joblib.load('dashboard/fake_news_model.pkl')
vectorizer = joblib.load('dashboard/tfidf_vectorizer.pkl')

# Function to clean user input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Text Input Box
news_text = st.text_area("Enter News Article Text:")

# Prediction Button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # Clean and Vectorize Input
        cleaned_input = clean_text(news_text)
        X_input = vectorizer.transform([cleaned_input])

        # Predict using trained model
        prediction = model.predict(X_input)

        # Display Result
        if prediction[0] == 0:
            st.success("✅ Real News Detected")
        else:
            st.error("🚫 Fake News Detected")
