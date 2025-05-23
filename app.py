import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App title
st.title("📰 Fake News Detector")

# Input text
input_text = st.text_area("Enter a news article text below 👇", height=200)

if st.button("Classify"):
    if input_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize input
        transformed_text = vectorizer.transform([input_text])
        # Predict
        prediction = model.predict(transformed_text)[0]
        # Show result
        if prediction == "FAKE":
            st.error("🛑 This news article is likely FAKE!")
        else:
            st.success("✅ This news article is likely REAL!")