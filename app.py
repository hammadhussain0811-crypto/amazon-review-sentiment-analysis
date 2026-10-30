import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon book review below to predict whether "
    "the sentiment is positive or negative."
)

review = st.text_area(
    "Enter your review:",
    placeholder="This book was amazing...",
    height=150
)

if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a review first.")

    elif len(review.strip()) < 3:
        st.warning("Please enter a longer review.")

    elif not any(char.isalpha() for char in review):
        st.warning("Your review must contain alphabetic characters.")

    else:
        try:
            response = requests.post(
                API_URL,
                json={"review": review}
            )

            response.raise_for_status()
            result = response.json()

            sentiment = result["sentiment"]

            if sentiment == "Positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("😞 Negative Sentiment")

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")