import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

# Define the dummy functions (same as in train.py)
def dummy_tokenizer(doc):
    return doc

def dummy_preprocessor(doc):
    return doc

# Preprocessing functions (same as in train.py)
def tokenize_text(text):
    """Tokenizes the input text."""
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    """Removes stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words and token not in string.punctuation]

def lemmatize_tokens(tokens):
    """Lemmatizes a list of tokens."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """Performs tokenization, stop-word removal, and lemmatization."""
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens

def main():
    """Main function to run inference using the trained sentiment classification model."""
    # Define paths
    test_data_path = os.path.join('/app/data', 'raw', 'final_project_test_dataset', 'test.csv')
    model_path = os.path.join('/app/outputs', 'models', 'model.pkl')
    predictions_path = os.path.join('/app/outputs', 'predictions', 'predictions.csv')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    # Load the trained model and vectorizer
    print("Loading trained model and vectorizer...")
    try:
        model, tfidf_vectorizer = joblib.load(model_path)  # Load both model and vectorizer
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure to run train.py first to train and save the model.")
        return

    # Load test data
    print("Loading test data...")
    try:
        test_df = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}")
        print("Please make sure to run data_loader.py first to download the data.")
        return

    # Preprocess the test data
    print("Preprocessing test data...")
    test_df['processed_review'] = test_df['review'].apply(preprocess_text)

    # Vectorize the test data using the loaded vectorizer
    print("Vectorizing test data...")
    X_test = tfidf_vectorizer.transform(test_df['processed_review'])

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Save predictions with 'review' and 'sentiment' columns
    predictions_df = pd.DataFrame({'review': test_df['review'], 'sentiment': y_pred})
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()