# 📈 FinSent - Financial Sentiment Analyzer


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

> 💹 A powerful financial sentiment analysis tool that leverages state-of-the-art machine learning models to analyze the sentiment of financial texts, news, and statements.

## ✨ Features

- 🤖 Multiple ML Models Support:
  - FinBERT (State-of-the-art transformer model)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
- 🎯 High Accuracy Predictions
- 💫 Modern & Responsive UI
- ⚡ Real-time Analysis
- 📊 Confidence Scores
- 🔄 Easy Model Switching

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FinSent.git
cd FinSent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python -m streamlit run app.py
```

4. Open your browser and navigate to:
```
http://localhost:8503
```

## 🎮 Usage

1. Enter your financial text in the input box
2. Select your preferred model:
   - **FinBERT** (Recommended): Specialized financial BERT model
   - **Logistic Regression**: Fast and reliable
   - **SVM**: Good for complex patterns
   - **Naive Bayes**: Efficient for text classification
3. Click "Analyze Sentiment" to get results
4. View the sentiment prediction, confidence score, and visual indicators

## 🛠️ Models

### FinBERT
- State-of-the-art transformer model
- Specifically trained on financial text
- Best accuracy for complex financial statements

### Traditional ML Models
- **Logistic Regression**: Linear classification
- **SVM**: Non-linear classification
- **Naive Bayes**: Probabilistic classification

## 🎨 UI Features

- Dark mode interface
- Responsive design
- Real-time analysis
- Interactive elements
- Progress indicators
- Sentiment-based icons
- Confidence visualization

## 📊 Example

```python
Input: "The company reported strong Q4 earnings, exceeding market expectations."
Output: {
    "prediction": "positive",
    "confidence": "0.9234",
    "model": "bert"
}
```

## 📸 Working Prototype

Here are some examples of FinSent in action:

### Positive Sentiment Analysis
![Positive Sentiment Analysis](sentiment-analysis/assets/Positive%20Sentement.png)


### Negative Sentiment Analysis
![Negative Sentiment Analysis](sentiment-analysis/assets/Negative%20Sentiment.png)


## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ProsusAI](https://github.com/ProsusAI/finbert) for the FinBERT model
- [Hugging Face](https://huggingface.co/) for transformer models
- [Flask](https://flask.palletsprojects.com/) for the web framework

---
Made with love for financial analysis
