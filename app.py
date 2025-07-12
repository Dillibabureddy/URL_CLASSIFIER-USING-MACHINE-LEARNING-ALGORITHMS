import streamlit as st
import joblib

# Load vectorizer and model using joblib
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("url_model.pkl")

st.set_page_config(page_title="URL Classification")
st.title("🔍 URL Classification")

url_input = st.text_input("Enter a URL")

if st.button("Predict"):
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        vectorized_url = vectorizer.transform([url_input])
        prediction = model.predict(vectorized_url)[0]
        st.success(f"Prediction: **{prediction.upper()}**")

        if hasattr(model, "predict_proba"):
            st.subheader("Prediction Probabilities:")
            probs = model.predict_proba(vectorized_url)[0]
            for label, prob in zip(model.classes_, probs):
                st.write(f"{label}: {prob:.2f}")
