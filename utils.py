"""
Utility functions for the Streamlit sentiment analysis application.

Provides model training/loading, text preprocessing, and prediction helpers
that mirror the pipeline in Phase3_Sentiment_Analysis.ipynb.
"""

import re

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import word_tokenize  # noqa: E402
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_PATH = "data/sentiment_results.csv"

# ── Data Loading ───────────────────────────────────────────────────────────────


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the sentiment results CSV produced by Phase 2."""
    df = pd.read_csv(path)
    df["Processed_Review"] = df["Processed_Review"].astype(str)
    return df


# ── Model Training ─────────────────────────────────────────────────────────────


def train_models(df: pd.DataFrame):
    """Train Word2Vec + LinearSVC and return all artefacts needed by the app.

    Returns a dict with keys:
        w2v_model, svc_model, calibrated_svc, label_encoder, class_names,
        review_vectors, tokenized_reviews, metrics, cv_scores,
        y_test, y_pred, confusion_mat
    """
    # Tokenise (column already cast to str in load_data)
    tokenized_reviews = [word_tokenize(r) for r in df["Processed_Review"]]

    # Word2Vec (CBOW, same hyper-params as Phase 3)
    w2v_model = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=100,
        window=5,
        min_count=2,
        sg=0,
        workers=4,
        epochs=50,
    )

    # Build review-level embeddings (mean pooling)
    review_vectors = np.array(
        [_get_review_vector(tokens, w2v_model) for tokens in tokenized_reviews]
    )

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df["lexicon_sentiment"])
    class_names = le.classes_

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        review_vectors, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # LinearSVC
    svc_model = LinearSVC(
        C=1.0, class_weight="balanced", max_iter=10000, random_state=42
    )
    svc_model.fit(X_train, y_train)

    # Calibrated wrapper for probability estimates
    calibrated_svc = CalibratedClassifierCV(svc_model, cv=5)
    calibrated_svc.fit(X_train, y_train)

    # Evaluate
    y_pred = svc_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=42),
        review_vectors,
        labels,
        cv=cv,
        scoring="f1_weighted",
    )

    return {
        "w2v_model": w2v_model,
        "svc_model": svc_model,
        "calibrated_svc": calibrated_svc,
        "label_encoder": le,
        "class_names": class_names,
        "review_vectors": review_vectors,
        "tokenized_reviews": tokenized_reviews,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "y_test": y_test,
        "y_pred": y_pred,
        "confusion_mat": cm,
    }


# ── Prediction Helpers ─────────────────────────────────────────────────────────


def preprocess_text(text: str) -> str:
    """Basic text cleaning – lowercase, strip non-alpha, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str, w2v_model, calibrated_svc, label_encoder):
    """Return (label, confidence, class_probabilities) for a single user string."""
    cleaned = preprocess_text(text)
    tokens = word_tokenize(cleaned)
    vec = _get_review_vector(tokens, w2v_model).reshape(1, -1)

    probas = calibrated_svc.predict_proba(vec)[0]
    pred_idx = np.argmax(probas)
    label = label_encoder.classes_[pred_idx]
    confidence = probas[pred_idx]

    class_probs = dict(zip(label_encoder.classes_, probas))
    return label, confidence, class_probs


# ── Internal Helpers ───────────────────────────────────────────────────────────


def _get_review_vector(tokens, model):
    """Mean-pool word vectors for one review."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)
