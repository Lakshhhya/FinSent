from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging
from pathlib import Path
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model paths
MODEL_DIR = Path("model")
MODELS = {
    "vectorizer": "tfidf_vectorizer.pkl",
    "label_encoder": "label_encoder.pkl",
    "logistic": "logistic_regression_baseline.pkl",
    "svc": "linear_svc_model.pkl",
    "nb": "multinomial_nb_model.pkl",
    "bert": "finbert-sentiment"
}

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

# Load models
try:
    vectorizer = joblib.load(MODEL_DIR / MODELS["vectorizer"])
    label_encoder = joblib.load(MODEL_DIR / MODELS["label_encoder"])
    logistic_model = joblib.load(MODEL_DIR / MODELS["logistic"])
    svc_model = joblib.load(MODEL_DIR / MODELS["svc"])
    nb_model = joblib.load(MODEL_DIR / MODELS["nb"])
    
    # Load or download FinBERT
    FINBERT_MODEL = "ProsusAI/finbert"
    bert_path = MODEL_DIR / MODELS["bert"]
    
    # Remove existing finbert directory if it exists but is incomplete
    if bert_path.exists():
        model_files = list(bert_path.glob('*'))
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json']
        if not all((bert_path / f).exists() for f in required_files):
            logger.info("Incomplete FinBERT model found. Removing and downloading fresh...")
            shutil.rmtree(bert_path)
        else:
            logger.info("Complete FinBERT model found.")
    
    if not bert_path.exists():
        logger.info(f"Downloading FinBERT model from {FINBERT_MODEL}...")
        # Create the directory
        bert_path.mkdir(exist_ok=True)
        
        # Download and save the model
        tokenizer_bert = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        bert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        
        # Save models locally
        tokenizer_bert.save_pretrained(str(bert_path))  # Convert Path to string
        bert_model.save_pretrained(str(bert_path))  # Convert Path to string
        logger.info("FinBERT model downloaded and saved successfully")
    else:
        logger.info("Loading FinBERT from local directory...")
        tokenizer_bert = AutoTokenizer.from_pretrained(str(bert_path))  # Convert Path to string
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_path))  # Convert Path to string
    
    bert_model.eval()  # Set to evaluation mode
    logger.info("All models loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def preprocess_text(text):
    """Clean and preprocess input text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_with_sklearn(model, vector, text):
    """Make predictions using scikit-learn models."""
    try:
        vec = vector.transform([text])
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(vec)[0]
        else:
            # For models without predict_proba (e.g., SVC)
            scores = model.decision_function(vec)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
        idx = np.argmax(probs)
        return idx, float(probs[idx])
    except Exception as e:
        logger.error(f"Error in sklearn prediction: {str(e)}")
        raise

def predict_with_bert(text):
    """Make predictions using FinBERT model."""
    try:
        inputs = tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])
    except Exception as e:
        logger.error(f"Error in BERT prediction: {str(e)}")
        raise

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get input data
        raw_text = request.form.get('text', '')
        model_name = request.form.get('model', '')

        # Validate input
        if not raw_text or not model_name:
            return jsonify({'error': 'Text and model selection are required'}), 400
        
        if model_name not in ['logistic', 'svc', 'nb', 'bert']:
            return jsonify({'error': f'Invalid model selection: {model_name}'}), 400

        # Preprocess text
        text = preprocess_text(raw_text)
        if not text:
            return jsonify({'error': 'Text contains no valid content after preprocessing'}), 400

        # Make prediction based on selected model
        try:
            if model_name == 'bert':
                idx, conf = predict_with_bert(raw_text)  # Use raw text for BERT
            else:
                model = {
                    'logistic': logistic_model,
                    'svc': svc_model,
                    'nb': nb_model
                }[model_name]
                idx, conf = predict_with_sklearn(model, vectorizer, text)

            sentiment = label_encoder.inverse_transform([idx])[0]
            
            return jsonify({
                'prediction': sentiment,
                'confidence': f"{conf:.4f}",
                'model': model_name
            })

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Error during prediction process'}), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
