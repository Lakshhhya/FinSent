# fin_sentiment_app.py

import streamlit as st
import pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# Load model and tokenizer
MODEL_PATH = "./sentiment-analysis/model/finbert-sentiment"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float32, 
    device_map=None,
    local_files_only=True,
    use_safetensors=True
)
model.eval()  # Set model to evaluation mode

# Predict function
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return ["Negative", "Neutral", "Positive"][predicted_class]

# Streamlit UI
st.set_page_config(page_title="FinSent - Financial Sentiment Analyzer", layout="centered")

st.title("📊 FinSent - Financial Sentiment Analyzer")
st.markdown("""
Analyze the sentiment of financial news, tweets, or statements in real-time using state-of-the-art NLP models.
""")

user_input = st.text_area("📝 Enter Financial Text:", placeholder="e.g. The company reported higher-than-expected earnings this quarter.")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.caption("Developed by Lakshya Dalal 🚀 | Powered by BERT")
