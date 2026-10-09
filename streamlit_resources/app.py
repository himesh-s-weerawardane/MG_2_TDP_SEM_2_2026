import streamlit as st
import pickle
import numpy as np

# Page title and prompt bar
st.title("Domain-Adaptive Sentiment & Public Opinion Analysis Demo")
st.write("Enter a prompt to predict its sentiment.")

# Collect inputs
# Accept user prompt
users_prompt = st.text_input("Please enter your Prompt", "e.g. Arizone Ice Tea Company share record profits...")
st.text(users_prompt)
# Select model option
st.write("Logistic Regression, SVM and Naive Bayes are traidtional Machine Learnign Models. BERT is a transformer model. Untrained is for Section 5")
user_selection = st.radio("Choose your model", ["Logistic Regression", "SVM", "Naive Bayes (Multinomial)", "BERT (trained)", "BERT (untrained)"])
st.write(user_selection)

# Identify the the model
if (user_selection == "Logistic Regression"):
    with open("pkls/models/lr_C.pkl", "rb") as f:
        model = pickle.load(f)   
    with open("pkls/vectorizer/lr_vectorizer_C.pkl", "rb") as f:
        vectorizer = pickle.load(f)
elif (user_selection == "SVM"):
    st.write(f">NOTE: The code needs to be adjusted to handle the proba function.")
    # with open("pkls/models/svm_C.pkl", "rb") as f:
    #     model = pickle.load(f)
    # with open("pkls/vectorizer/svm_vectorizer_C.pkl", "rb") as f:
    #     vectorizer = pickle.load(f)
elif (user_selection == "Naive Bayes (Multinomial)"):
    with open("pkls/models/nb_C.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pkls/vectorizer/nb_vectorizer_C.pkl", "rb") as f:
        vectorizer = pickle.load(f)
elif (user_selection == "BERT (trained)"):
    st.write(f">NOTE: Awaiting implementation")
#     with open("bert_t.pkl", "rb") as f:
#         model = pickle.load(f)
elif (user_selection == "BERT (untrained)"):
    st.write(f">NOTE: Awaiting implementation")
#     with open("bert_ut.pkl", "rb") as f:
        # model = pickle.load(f)

# Predict model the sentimetn.
if st.button("Predict Sentiment"):
    vectorized = vectorizer.transform([users_prompt])
    probability = model.predict_proba(vectorized)[0][1]
    prediction = model.predict(vectorized)[0]
    print("Mode rpedict", prediction, " model probability", probability)

    st.subheader("Prediction Result")
    st.write(f"Probability: **{probability:.2%}**")
    if (prediction == 1):
        st.success("Model predicts that this is POSITIVE (BULLISH)")
    else:
        st.warning("Model predicts that this is NEGATIVE (BEARISH)")

    # File to store test prompts and data
    with open("streamlit_usage.txt", "a") as f:
        f.write(f"> Prompt: {users_prompt}\n")
        f.write(f"Prediction: {prediction}\n")
        f.write(f"Probability: {probability}\n")
        f.write(f"Model: {user_selection}\n\n")
