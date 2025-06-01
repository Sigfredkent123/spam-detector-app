# streamlit_app.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📩 Spam Message Classifier")
st.write("Enter a message to see whether it's spam or not:")

# Input box
user_input = st.text_area("Your message", height=100)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform and predict
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)

        label = "Spam" if prediction[0] == 1 else "Ham"
        st.success(f"Prediction: **{label}**")
