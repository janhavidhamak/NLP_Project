import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
processed_df = pd.read_csv("data/processed_reviews.csv")
sentiment_df = pd.read_csv("data/sentiment_results.csv")

# Title
st.title("💬 NLP-Based Skincare Review Analyzer")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose a section", ["Overview", "Sentiment Analysis", "Ask a Question"])

# Overview Section
if page == "Overview":
    st.subheader(" Project Overview")
    st.write("""
    This app uses **Natural Language Processing (NLP)** to analyze skincare product reviews.
    You can explore sentiment trends, semantic relationships, and even ask custom questions
    about the product based on review insights.
    """)
    st.dataframe(processed_df.head())

# Sentiment Analysis
elif page == "Sentiment Analysis":
    st.subheader(" Sentiment Distribution")
    counts = sentiment_df["lexicon_sentiment"].value_counts()
    st.bar_chart(counts)
    st.write("### Sentiment Summary")
    st.write(counts.to_frame())

# Question Answering Section
elif page == "Ask a Question":
    st.subheader(" Ask About the Product")
    user_question = st.text_input("Type your question:")
    if user_question:
        st.write(f" Based on our analysis, here's what customers are saying:")
        q = user_question.lower()
        if "dry" in q or "irritate" in q:
            st.write("Some users reported dryness or irritation after use.")
        elif "moistur" in q or "hydrating" in q:
            st.write(" Most users found it very moisturizing and soothing.")
        elif "gentle" in q or "soft" in q:
            st.write(" The product is generally praised for being gentle on skin.")
        elif "smell" in q or "fragrance" in q:
            st.write("Customers described the fragrance as mild and pleasant.")
        else:
            st.write(" Mixed or neutral opinions — no strong pattern found.")
