import streamlit as st
import joblib
import sys
import os
import nltk


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import clean_text

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Financial News Sentiment CLassifier")
st.write("Enter a financial news headline or sentence to predict its sentiment.")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

def clear_all():
    st.session_state.user_input = ""
    st.session_state.prediction_result = None

user_input = st.text_area("News text", height = 100, key = "user_input")

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Predict Sentiment")
with col2:
    st.button("Refresh", on_click=clear_all)

if predict_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        classes = model.classes_
        st.session_state.prediction_result = {
            "prediction": prediction,
            "proba": dict(zip(classes, proba))
        }

if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    st.subheader(f"Predicted Sentiment: {result['prediction'].capitalize()}")
    st.write("Confidence breakdown:")
    for cls, p in result["proba"].items():
        st.write(f"{cls.capitalize()}: {p*100:.1f}%")


