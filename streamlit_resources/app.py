import streamlit as st
import pickle
import numpy as np
import os
import re
import emoji
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# BERT models (from seed 42) published on the Hugging Face
BERT_STRATEGIES = {
    "BERT - Strategy 1 (finance only)": {
        "repo": "sandinator/bert-stockemotions-domain-s1",
        "description": "bert-base-uncased fine-tuned on StockEmotions only (8,000 finance posts)",
        "accuracy": 0.786, "macro_f1": 0.784},
    "BERT - Strategy 2 (general then finance)": {
        "repo": "sandinator/bert-stockemotions-sequential-s2",
        "description": "fine-tuned on TweetEval first, then continued on StockEmotions (sequential transfer)",
        "accuracy": 0.782, "macro_f1": 0.781},
    "BERT - Strategy 3 (mixed)": {
        "repo": "sandinator/bert-stockemotions-mixed-s3",
        "description": "fine-tuned once on TweetEval + StockEmotions combined (32,933 posts)",
        "accuracy": 0.775, "macro_f1": 0.774},
}

# (Extra Enhancement) Second BERT Emotion Model with 12 outputs published on the Hugging Face
EMOTION_MODEL_ID = "sandinator/bert-stockemotions-emotion"

# Results folder (bert_results.csv + images), relative to this file.
BERT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "transformer based modelling", "results")

# Function: To prepare the text for BERT prediction including fixing literal escape codes and converting emojis to words.
def prepare_text(text):
    text = str(text)
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()

# Function: To load one BERT model once (per Hub name) and keeps it in memory for later predictions.
@st.cache_resource
def load_bert(repo_id):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

# Function: Predicts the sentiment of a given text using the chosen BERT model.
def bert_predict(text, repo_id):
    tokenizer, model = load_bert(repo_id)
    inputs = tokenizer(prepare_text(text), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    return int(probabilities.argmax()), float(probabilities[1])

# Function: Loads the emotion model once.
@st.cache_resource
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
    model.eval()
    return tokenizer, model

# Function: To predict the emotion of a given text using the loaded emotion model.
def emotion_predict(text):
    tokenizer, model = load_emotion_model()
    inputs = tokenizer(prepare_text(text), return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]
    all_probabilities = {model.config.id2label[i]: float(p) for i, p in enumerate(probabilities)}
    best = max(all_probabilities, key=all_probabilities.get)
    return best, all_probabilities[best], all_probabilities


# Page Title and Prompt Bar
st.title("Domain-Adaptive Sentiment & Public Opinion Analysis Demo")
st.write("Enter a prompt to predict its sentiment.")

# User Input Collection
users_prompt = st.text_input("Please enter your Prompt", "e.g. Arizone Ice Tea Company share record profits...")
st.text(users_prompt)

# Model Selection
st.write("Logistic Regression, SVM and Naive Bayes are traidtional Machine Learnign Models. BERT is a transformer model.")
user_selection = st.radio("Choose your model",
                          ["Logistic Regression", "SVM", "Naive Bayes (Multinomial)"] + list(BERT_STRATEGIES) + ["BERT (untrained)"])
st.write(user_selection)

# Match the User Selection
if (user_selection == "Logistic Regression"):
    with open("pkls/models/lr_C.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pkls/vectorizer/lr_vectorizer_C.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    accuracy = 0.707
elif (user_selection == "SVM"):
    with open("pkls/models/svm_C.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pkls/vectorizer/svm_vectorizer_C.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    accuracy = 0.709
elif (user_selection == "Naive Bayes (Multinomial)"):
    with open("pkls/models/nb_C.pkl", "rb") as f:
        model = pickle.load(f)
    with open("pkls/vectorizer/nb_vectorizer_C.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    accuracy = 0.637
elif user_selection in BERT_STRATEGIES:
    bert_info = BERT_STRATEGIES[user_selection]
    accuracy = bert_info["accuracy"]
    macro_f1 = bert_info["macro_f1"]
    # Info Cards
    left, right = st.columns(2)
    with left:
        st.info(f"**Sentiment model** ({user_selection.replace('BERT - ', '')})\n\n"
                f"{bert_info['description']}\n\n"
                f"Test accuracy **{accuracy:.1%}** | macro-F1 **{macro_f1:.3f}**")
    with right:
        st.info("**Emotion model** (Extra Enhancement)\n\n"
                "12-class BERT trained on StockEmotions emotion labels\n\n"
                "Test accuracy **41.8%**")
elif (user_selection == "BERT (untrained)"):
    st.write(f">NOTE: Awaiting implementation")
#     with open("bert_ut.pkl", "rb") as f:
        # model = pickle.load(f)

# Predict model the sentiment.
if st.button("Predict Sentiment"):
    if (user_selection == "BERT (untrained)"):
        st.error("BERT (untrained) is not implemented yet.")
        st.stop()

    st.subheader("Prediction Result")
    if user_selection in BERT_STRATEGIES:
        with st.spinner("Running BERT..."):
            prediction, probability = bert_predict(users_prompt, bert_info["repo"])
            emotion_name, emotion_probability, all_emotions = emotion_predict(users_prompt)
        sentiment_name = "bullish" if prediction == 1 else "bearish"
        sentiment_probability = probability if prediction == 1 else 1 - probability

        # 2 Title metrics for sentiment and emotion
        left, right = st.columns(2)
        left.metric("Sentiment", sentiment_name.upper(), f"{sentiment_probability:.0%} confidence",
                    delta_color="normal" if prediction == 1 else "inverse")
        right.metric("Emotion", emotion_name.capitalize(), f"{emotion_probability:.0%} confidence", delta_color="normal")
        st.caption(f"Sentiment model: test accuracy {accuracy:.1%}, macro-F1 {macro_f1:.3f}.")
        st.caption("Emotion model: test accuracy 41.8% (demo extra).")

        # Top 5 emotions as a bar chart
        st.markdown("##### Top 5 emotions")
        top_emotions = pd.Series(all_emotions).sort_values(ascending=False).head(5)
        st.bar_chart(top_emotions, horizontal=True, height=220)
    else:
        vectorized = vectorizer.transform([users_prompt])
        prediction = model.predict(vectorized)[0]
        if (user_selection != "SVM"):
            probability = model.predict_proba(vectorized)[0][1]
            st.write(f"Probability: **{probability:.2%}**")
        else:
            probability = "NOT APPLICABLE"
        st.write(f"Accuracy: **{accuracy:.2%}** (from Model C during testing)")

    if (prediction == 1):
        st.success("Model predicts that this is POSITIVE (BULLISH)")
    else:
        st.warning("Model predicts that this is NEGATIVE (BEARISH)")

    # File to store test prompts and data
    with open("streamlit_usage.txt", "a") as f:
        f.write(f"> Prompt: {users_prompt}\n")
        f.write(f"Prediction: {prediction}\n")
        f.write(f"Probability: {probability}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Model: {user_selection}\n\n")



# Findings - BERT training strategies compared on the StockEmotions test set.
st.divider()
with st.expander("Findings on BERT training strategies (3 seeds each)"):
    results_file = os.path.join(BERT_RESULTS_DIR, "bert_results.csv")
    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
        strategy_names = {
            "A_general_only": "Model A: general tweets only",
            "S1_domain_only": "Strategy 1: finance only",
            "S2_sequential": "Strategy 2: general then finance",
            "S3_mixed": "Strategy 3: mixed",
        }
        summary = results.groupby("strategy")[["accuracy", "macro_f1", "bearish_recall"]].agg(["mean", "std"]).round(3)
        summary.columns = [f"{metric} ({stat})" for metric, stat in summary.columns]
        summary.index = [strategy_names.get(s, s) for s in summary.index]
        st.write("Mean and standard deviation over seeds 42, 43 and 44 (StockEmotions test set, 1,000 posts).")
        st.dataframe(summary)
        st.write("Reference: best TF-IDF baseline (Logistic Regression, Model B) macro-F1 0.711; always predicting bullish 0.357.")

        comparison_chart = os.path.join(BERT_RESULTS_DIR, "images", "bert_strategy_comparison.png")
        if os.path.exists(comparison_chart):
            st.image(comparison_chart)
        for strategy, name in strategy_names.items():
            picture = os.path.join(BERT_RESULTS_DIR, "images", f"{strategy}_seed42.png")
            if os.path.exists(picture):
                st.image(picture, caption=f"{name} (seed 42)", width=360)
    else:
        st.info("Results file not found: add src/transformer based modelling/results/bert_results.csv to the repository.")
