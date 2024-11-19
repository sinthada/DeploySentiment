import pickle
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# Load the saved model
with open('sentiment_pipeline_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("Sentiment Analysis")

# Input box for user to type a sentence
user_input = st.text_input("Enter a sentece:")

#Predict sentiment when user submits input
if st.button("Predict Sentiment"):
    if user_input:
        prediction = loaded_model.predict([user_input])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a sentence to analyze .")
