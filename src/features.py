from pre_processing import preprocess_text
#print(preprocess_text)  # Example usage

import spacy
from spacy import displacy
import pandas as pd

# Load model
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_csv("data/processed_reviews.csv")
print(df.head(10))

for i, text in enumerate(df["processed_text"].dropna().head(5), 1):
    print(f"\n🧾 Review {i}: {text}")
    doc = nlp(str(text))
    displacy.serve(doc, style="dep", port=5000)
    break 
