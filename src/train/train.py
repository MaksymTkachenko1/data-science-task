import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import numpy as np
import re  # <-- added import for regex operations

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ensure NLTK data is available
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

# Define dummy functions for the vectorizer
def dummy_tokenizer(doc):
    return doc

def dummy_preprocessor(doc):
    return doc

# Preprocessing functions
def clean_text(text):
    """Removes URLs, HTML tags, numbers, and non-ASCII characters from text."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)            # Remove HTML tags
    text = re.sub(r'\d+', '', text)              # Remove numbers
    text = re.sub(r'[^\x00-\x7F]+','', text)      # Remove non-ASCII characters (e.g., emojis)
    return text

def tokenize_text(text):
    """Tokenizes the input text."""
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    """Removes stopwords (except 'not') and punctuation from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    stop_words.discard("not")  # Retain 'not' for sentiment analysis
    return [token for token in tokens if token not in stop_words and token not in string.punctuation]

def lemmatize_tokens(tokens):
    """Lemmatizes a list of tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """Cleans and processes text: cleaning, tokenization, stop-word removal, and lemmatization."""
    text = clean_text(text)  # <-- clean the text first
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens

def main():
    """Main function to train the sentiment classification model."""
    # Define paths
    train_data_path = os.path.join('/app/data', 'raw', 'final_project_train_dataset', 'train.csv')
    test_data_path = os.path.join('/app/data', 'raw', 'final_project_test_dataset', 'test.csv')
    model_path = os.path.join('/app/outputs', 'models', 'model.pkl')
    metrics_path = os.path.join('/app/outputs', 'predictions', 'metrics.txt')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Preprocess data
    print("Preprocessing data...")
    train_df['processed_review'] = train_df['review'].apply(preprocess_text)
    test_df['processed_review'] = test_df['review'].apply(preprocess_text)

    # Vectorize data
    print("Vectorizing data...")
    tfidf_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer, preprocessor=dummy_preprocessor, lowercase=False, token_pattern=None)
    X_train = tfidf_vectorizer.fit_transform(train_df['processed_review'])
    X_test = tfidf_vectorizer.transform(test_df['processed_review'])
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    # Train the model
    print("Training model...")
    model = LinearSVC(max_iter=2000, dual=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Save the model and the vectorizer
    joblib.dump((model, tfidf_vectorizer), model_path)  # Save both model and vectorizer
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")

    print(f"Model and vectorizer saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()