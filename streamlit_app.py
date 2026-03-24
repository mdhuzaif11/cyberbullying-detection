import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("LinearSVC.pkl", "rb"))
vectorizer = pickle.load(open("tfidfvectorizer.pkl", "rb"))

st.title("Cyberbullying Detection System")

st.write("Enter a comment to check if it is cyberbullying.")

user_input = st.text_area("Enter Text")

if st.button("Predict"):

    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)[0]

    if prediction == 1:
        st.error("Cyberbullying Detected")
    else:
        st.success("No Cyberbullying Detected")