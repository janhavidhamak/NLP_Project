import re
import pandas as pd
import nltk
from langdetect import detect
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import langcodes

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
translator = Translator()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def translate_to_english(text):
    try:
        translated = translator.translate(text, src='auto', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

# Load dataset
df = pd.read_csv('data/product_reviews.csv')

# Preprocess
df['preprocessed_text'] = df['review_text'].apply(preprocess_text)

# Detect language and convert to full name
df['lang_code'] = df['preprocessed_text'].apply(lambda x: detect(str(x)))
df['language'] = df['lang_code'].apply(lambda code: langcodes.Language.get(code).language_name())

# Translate
df['translated_text'] = df['preprocessed_text'].apply(translate_to_english)

# Tokenize + Lemmatize
df['processed_text'] = df['translated_text'].apply(tokenize_and_lemmatize)

# Save
df.to_csv('data/processed_reviews.csv', index=False, encoding='utf-8')

# Preview
print(df[['date', 'review_text', 'language', 'translated_text', 'processed_text']].head())
