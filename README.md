# NLP-Based Skincare Product Review Analysis

This project applies **Natural Language Processing (NLP)** to skincare product reviews collected from Amazon.  
It aims to analyze customer sentiment, identify important product features, and explore semantic relationships between terms.  
The project covers **data preprocessing, POS tagging, Named Entity Recognition (NER), Bag-of-Words, TF-IDF**, and **Word2Vec embeddings**.

---


Key Insights
Most frequent positive words: gentle, soft, refreshing, moisturizing
Common negative mentions: dry, irritation, breakouts
Adjectives dominate the reviews, indicating a focus on texture and feel
TF-IDF and Word2Vec highlight core themes like hydration and skin smoothness

## Running the Streamlit App

```bash
# 1. Clone and install dependencies
git clone https://github.com/janhavidhamak/NLP_Project.git
cd NLP_Project
pip install -r requirements.txt

# 2. Launch the application
streamlit run app.py
```

The app opens at **http://localhost:8501** and provides three tabs:

| Tab | Description |
|-----|-------------|
| 📊 Data Analysis | Sentiment distribution charts, confusion matrix, performance metrics, Lexicon vs LinearSVC comparison |
| 🔍 Sentiment Predictor | Enter any review text and get a real-time sentiment prediction with confidence scores |
| 📈 Model Insights | Embedding dimension weights, top sentiment words, cross-validation results, model configuration |

> On the first run the models are trained from `data/sentiment_results.csv` and cached automatically — subsequent page loads are instant.

