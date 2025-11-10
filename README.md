# NLP-Based Skincare Product Review Analysis

This project applies **Natural Language Processing (NLP)** to skincare product reviews collected from Amazon.  
It aims to analyze customer sentiment, identify important product features, and explore semantic relationships between terms.  
The project covers **data preprocessing, POS tagging, Named Entity Recognition (NER), Bag-of-Words, TF-IDF**, and **Word2Vec embeddings**.

---

## Project Structure
NLP_Project/
│
├── data/
│ ├── amazon_reviews.csv # Original dataset
│ ├── processed_reviews.csv # Cleaned + preprocessed data
│ ├── bow_features.csv # Bag-of-Words feature matrix
│ ├── tfidf_matrix.npz # TF-IDF vectorized matrix
│ ├── tfidf_word_similarity.csv # Word similarity (TF-IDF)
│ ├── sentiment_results.csv # Lexicon-based sentiment output
│ └── word2vec_skincare.bin # Word2Vec model file
│
│
├── Phase1.ipynb # Data preprocessing
├── Phase2.ipynb # POS tagging, NER, BoW, TF-IDF, Word2Vec,Review Summarization using TF-IDF Similarity
├── Scrapping.ipynb # Review scraping script
│
├── requirements.txt # Project dependencies
└── README.md # Documentation


Key Insights
Most frequent positive words: gentle, soft, refreshing, moisturizing
Common negative mentions: dry, irritation, breakouts
Adjectives dominate the reviews, indicating a focus on texture and feel
TF-IDF and Word2Vec highlight core themes like hydration and skin smoothness

