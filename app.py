"""
Streamlit application for NLP Sentiment Analysis of Skincare Reviews.

Run with:  streamlit run app.py
"""

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_data, predict_sentiment, train_models

# ── NLTK bootstrap (one-time) ─────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Sentiment Analysis",
    page_icon="🧴",
    layout="wide",
)

# ── Colours / constants ───────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral": "#95a5a6",
    "negative": "#e74c3c",
}


# ── Cached model loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models – this only happens once …")
def get_models():
    """Load data, train all models, and cache the results."""
    df = load_data()
    artefacts = train_models(df)
    return df, artefacts


df, artefacts = get_models()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🧴 Skincare Sentiment")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df)} reviews")
st.sidebar.markdown(
    f"**Model accuracy:** {artefacts['metrics']['accuracy']:.1%}"
)
st.sidebar.markdown(
    f"**CV F1 (mean):** {artefacts['cv_scores'].mean():.4f}"
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Phase 3 NLP Project")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_analysis, tab_predict, tab_insights = st.tabs(
    ["📊 Data Analysis", "🔍 Sentiment Predictor", "📈 Model Insights"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.header("Training Data Analysis")

    # ── Sentiment distribution ──────────────────────────────────────────────
    st.subheader("Sentiment Distribution")
    col_pie, col_bar = st.columns(2)

    dist = df["lexicon_sentiment"].value_counts()
    with col_pie:
        fig_pie = px.pie(
            names=dist.index,
            values=dist.values,
            color=dist.index,
            color_discrete_map=SENTIMENT_COLORS,
            title="Lexicon-Based Sentiment",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            x=dist.index,
            y=dist.values,
            color=dist.index,
            color_discrete_map=SENTIMENT_COLORS,
            labels={"x": "Sentiment", "y": "Count"},
            title="Lexicon-Based Sentiment Counts",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Confusion matrix ────────────────────────────────────────────────────
    st.subheader("Confusion Matrix")
    cm = artefacts["confusion_mat"]
    class_names = list(artefacts["class_names"])

    fig_cm = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="LinearSVC Confusion Matrix (Test Set)",
    )
    fig_cm.update_layout(width=500, height=450)
    st.plotly_chart(fig_cm, use_container_width=False)

    # ── Performance metrics ─────────────────────────────────────────────────
    st.subheader("Model Performance Metrics")
    m = artefacts["metrics"]
    cols = st.columns(4)
    for i, (name, val) in enumerate(m.items()):
        cols[i].metric(name.capitalize(), f"{val:.4f}")

    # ── Lexicon vs LinearSVC comparison ─────────────────────────────────────
    st.subheader("Sentiment Comparison: Lexicon vs LinearSVC")

    svc_model = artefacts["svc_model"]
    le = artefacts["label_encoder"]
    review_vectors = artefacts["review_vectors"]

    all_pred_labels = le.inverse_transform(svc_model.predict(review_vectors))
    svc_dist = pd.Series(all_pred_labels).value_counts()

    comparison = pd.DataFrame(
        {"Lexicon-Based": dist, "LinearSVC": svc_dist}
    ).fillna(0).astype(int)

    fig_comp = go.Figure()
    for method in comparison.columns:
        fig_comp.add_trace(
            go.Bar(
                name=method,
                x=comparison.index,
                y=comparison[method],
            )
        )
    fig_comp.update_layout(
        barmode="group",
        title="Sentiment Distribution – Lexicon vs LinearSVC",
        xaxis_title="Sentiment",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    agreement = (all_pred_labels == df["lexicon_sentiment"].values).mean()
    st.info(f"Agreement between Lexicon and LinearSVC methods: **{agreement:.1%}**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Sentiment Predictor
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Real-Time Sentiment Predictor")
    st.markdown("Enter a skincare product review and the model will predict its sentiment.")

    user_text = st.text_area(
        "Type your review here:",
        height=120,
        placeholder="e.g. This moisturizer made my skin feel soft and hydrated …",
    )

    if st.button("Analyse Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            label, confidence, class_probs = predict_sentiment(
                user_text,
                artefacts["w2v_model"],
                artefacts["calibrated_svc"],
                artefacts["label_encoder"],
            )

            # Colour-coded result
            colour = SENTIMENT_COLORS.get(label, "#95a5a6")
            st.markdown(
                f"### Predicted Sentiment: "
                f'<span style="color:{colour}; font-weight:bold">{label.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {confidence:.1%}")

            # Probability breakdown
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame(
                {
                    "Sentiment": list(class_probs.keys()),
                    "Probability": list(class_probs.values()),
                }
            )
            fig_prob = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                title="Prediction Confidence Breakdown",
            )
            fig_prob.update_layout(showlegend=False, yaxis_range=[0, 1])
            st.plotly_chart(fig_prob, use_container_width=True)

    # ── Sample predictions ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Sample Predictions from Dataset")

    calibrated_svc = artefacts["calibrated_svc"]
    all_confidence = np.max(calibrated_svc.predict_proba(review_vectors), axis=1)

    sample = pd.DataFrame(
        {
            "Review": df["Processed_Review"].str[:80] + " …",
            "Lexicon": df["lexicon_sentiment"],
            "LinearSVC": all_pred_labels,
            "Confidence": all_confidence.round(3),
        }
    ).head(15)
    st.dataframe(sample, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.header("Model Insights")

    # ── Feature importance ──────────────────────────────────────────────────
    st.subheader("Embedding Dimension Weights")

    # coef_[0] is the separating hyperplane for binary classification
    # (the dataset has two classes: neutral and positive)
    coef = svc_model.coef_[0]
    dim_df = pd.DataFrame(
        {"Dimension": np.arange(len(coef)), "Weight": coef}
    )
    dim_df["Direction"] = np.where(dim_df["Weight"] > 0, "Positive", "Negative")

    fig_weights = px.bar(
        dim_df,
        x="Dimension",
        y="Weight",
        color="Direction",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
        title="LinearSVC Weights Across 100 Word2Vec Dimensions",
    )
    fig_weights.update_layout(showlegend=True)
    st.plotly_chart(fig_weights, use_container_width=True)

    # ── Top positive & negative dimensions ──────────────────────────────────
    st.subheader("Top Sentiment Dimensions")
    n_top = 15
    top_pos = np.argsort(coef)[-n_top:][::-1]
    top_neg = np.argsort(coef)[:n_top]

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**🟢 Top Positive Dimensions**")
        pos_df = pd.DataFrame(
            {"Dimension": top_pos, "Weight": coef[top_pos]}
        )
        fig_pos = px.bar(
            pos_df,
            x="Dimension",
            y="Weight",
            color_discrete_sequence=["#2ecc71"],
            title="Top Positive Dimensions",
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with col_neg:
        st.markdown("**🔴 Top Negative Dimensions**")
        neg_df = pd.DataFrame(
            {"Dimension": top_neg, "Weight": coef[top_neg]}
        )
        fig_neg = px.bar(
            neg_df,
            x="Dimension",
            y="Weight",
            color_discrete_sequence=["#e74c3c"],
            title="Top Negative Dimensions",
        )
        st.plotly_chart(fig_neg, use_container_width=True)

    # ── Words associated with top dimensions ────────────────────────────────
    st.subheader("Words Associated with Top Sentiment Dimensions")

    w2v = artefacts["w2v_model"]
    vocab_words = w2v.wv.index_to_key
    word_matrix = np.array([w2v.wv[w] for w in vocab_words])

    for label_type, dims in [("Positive", top_pos[:5]), ("Negative", top_neg[:5])]:
        colour = "#2ecc71" if label_type == "Positive" else "#e74c3c"
        st.markdown(f"**{label_type} sentiment dimensions**")
        rows = []
        for dim in dims:
            dim_scores = word_matrix[:, dim]
            top_indices = np.argsort(np.abs(dim_scores))[-5:][::-1]
            words = ", ".join(
                f"{vocab_words[i]} ({dim_scores[i]:.3f})" for i in top_indices
            )
            rows.append({"Dimension": int(dim), "Weight": f"{coef[dim]:+.4f}", "Top Words": words})
        st.table(pd.DataFrame(rows))

    # ── Cross-validation results ────────────────────────────────────────────
    st.subheader("5-Fold Cross-Validation")
    cv_scores = artefacts["cv_scores"]
    cv_df = pd.DataFrame(
        {"Fold": [f"Fold {i+1}" for i in range(len(cv_scores))], "F1 Score": cv_scores}
    )
    fig_cv = px.bar(
        cv_df,
        x="Fold",
        y="F1 Score",
        title="Cross-Validation F1 Scores",
        color_discrete_sequence=["#3498db"],
    )
    fig_cv.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_cv, use_container_width=True)

    st.metric("Mean CV F1", f"{cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    # ── Model configuration ─────────────────────────────────────────────────
    st.subheader("Model Configuration")
    st.json(
        {
            "Classifier": "LinearSVC (C=1.0, class_weight='balanced')",
            "Features": "Word2Vec embeddings (100 dimensions)",
            "Word2Vec": "CBOW, window=5, min_count=2, epochs=50",
            "Dataset": f"{len(df)} reviews",
            "Train/Test Split": "80/20 (stratified)",
        }
    )
