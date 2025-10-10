# Data Libs
import json
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import Tuple
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
# spacy_model = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # ----------------- 1. Preprocessing Pipeline --------------------
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)



def time_train_val_test_split(df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a time-series DataFrame into training, validation, and test sets.

    Args:
        df: The input DataFrame, which must be sorted by time.
        train_frac: The fraction of the data to use for the training set.
        val_frac: The fraction of the data to use for the validation set.

    Returns:
        A tuple containing the training, validation, and test DataFrames.
    """
    n = len(df)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)

    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size : train_size + val_size].copy()
    test = df.iloc[train_size + val_size:].copy()

    return train, val, test

# -----------------------
# Load + Preprocess Data
# -----------------------
def load_data(news_data, test_size=0.2, val_size=0.1):
    df = pd.read_csv(news_data)
    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['date'], utc=True, format='mixed')
    # Parse sentiment dict into columns
    df['sentiment_class'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'", '"'))['class'])
    df['polarity'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'", '"'))['polarity'])
    df['subjectivity'] = df['sentiment'].apply(lambda x: json.loads(x.replace("'", '"'))['subjectivity'])
    # Combine text
    df['content'] = df['title'] + " " + df['text']
    # Choose features
    df.dropna(subset=['content', 'sentiment_class'], inplace=True)
    # Filter for binary classification (negative and positive)
    df = df[df['sentiment_class'].isin(['negative', 'neutral', 'positive'])]
    df['clean_text'] = df['content'].map(clean_text)
    df = df.sort_index(ascending=True)

    return time_train_val_test_split(df)