import streamlit as st
import pickle

st.title("Cyberbullying Detection System")

st.write("This system detects whether a comment contains cyberbullying using Machine Learning models.")

# Load vectorizer
vectorizer = pickle.load(open("tfidfvectorizer.pkl","rb"))

# Model selection
model_name = st.selectbox(
    "Select Machine Learning Model",
    (
        "Linear SVC",
        "Logistic Regression",
        "Naive Bayes",
        "Decision Tree",
        "AdaBoost",
        "Bagging",
        "SGD Classifier"
    )
)

# Load selected model
if model_name == "Linear SVC":
    model = pickle.load(open("LinearSVC.pkl","rb"))

elif model_name == "Logistic Regression":
    model = pickle.load(open("LogisticRegression.pkl","rb"))

elif model_name == "Naive Bayes":
    model = pickle.load(open("MultinomialNB.pkl","rb"))

elif model_name == "Decision Tree":
    model = pickle.load(open("DecisionTreeClassifier.pkl","rb"))

elif model_name == "AdaBoost":
    model = pickle.load(open("AdaBoostClassifier.pkl","rb"))

elif model_name == "Bagging":
    model = pickle.load(open("BaggingClassifier.pkl","rb"))

elif model_name == "SGD Classifier":
    model = pickle.load(open("SGDClassifier.pkl","rb"))


# User input
text = st.text_area("Enter a Comment")

# Prediction
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        transform = vectorizer.transform([text])
        prediction = model.predict(transform)

        if prediction[0] == 1:
            st.error("Cyberbullying Detected")

        else:
            st.success("No Cyberbullying Detected")