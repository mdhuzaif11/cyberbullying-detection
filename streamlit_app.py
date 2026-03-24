import streamlit as st
import pickle
import os

st.set_page_config(page_title="Cyberbullying Detection", page_icon="💬")

st.title("Cyberbullying Detection System")
st.write("Detect whether a comment contains cyberbullying using Machine Learning models.")

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load vectorizer
vectorizer_path = os.path.join(BASE_DIR, "tfidfvecotizer.pkl")
with open(vectorizer_path, "rb") as f:
    vectoizer = pickle.load(f)

# Model selection
model_name = st.selectbox(
    "Select Model",
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

# Load model
if model_name == "Linear SVC":
    model_file = "LinearSVC.pkl"

elif model_name == "Logistic Regression":
    model_file = "LogisticRegression.pkl"

elif model_name == "Naive Bayes":
    model_file = "MultinomialNB.pkl"

elif model_name == "Decision Tree":
    model_file = "DecisionTreeClassifier.pkl"

elif model_name == "AdaBoost":
    model_file = "AdaBoostClassifier.pkl"

elif model_name == "Bagging":
    model_file = "BaggingClassifier.pkl"

elif model_name == "SGD Classifier":
    model_file = "SGDClassifier.pkl"

model_path = os.path.join(BASE_DIR, model_file)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# User input
text = st.text_area("Enter Comment")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        transformed_text = vectoizer.transform([text])
        prediction = model.predict(transformed_text)

        if prediction[0] == 1:
            st.error("Cyberbullying Detected")
        else:
            st.success("No Cyberbullying Detected")