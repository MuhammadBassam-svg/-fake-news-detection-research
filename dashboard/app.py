import streamlit as st
import pandas as pd
import joblib

# Title
st.title("📰 Fake News Detector")

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text input
news_text = st.text_area("Enter News Article Text:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize and predict
        X_input = vectorizer.transform([news_text])
        prediction = model.predict(X_input)

        # Output result
        if prediction[0] == 0:
            st.success("✅ Real News Detected")
        else:
            st.error("🚫 Fake News Detected")
