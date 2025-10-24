"""
NLTK utilities for text processing in the chatbot.
Handles tokenization, stemming, and bag-of-words conversion.
"""

import nltk
import numpy as np
import json
import pickle
import os
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# -----------------------------------------------------------
# Download required NLTK data (only once)
# -----------------------------------------------------------
def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

# Download NLTK data once at import
download_nltk_data()

# Initialize stemmer
stemmer = PorterStemmer()

# -----------------------------------------------------------
# Tokenization and Preprocessing
# -----------------------------------------------------------
def tokenize(sentence):
    """
    Tokenize a sentence into words.

    Args:
        sentence (str): Input sentence to tokenize

    Returns:
        list: List of tokenized words (lowercased and cleaned)
    """
    # Convert to lowercase and remove punctuation
    sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))

    # Tokenize using NLTK
    tokens = nltk.word_tokenize(sentence)
    return tokens


def stem(word):
    """
    Stem a word using Porter Stemmer.

    Args:
        word (str): Word to stem

    Returns:
        str: Stemmed word
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag of words representation.

    Args:
        tokenized_sentence (list): List of tokenized words
        all_words (list): List of all unique words in the vocabulary

    Returns:
        numpy.ndarray: Bag of words array
    """
    # Lowercase and stem words
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Initialize bag with zeros
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Mark the presence of words
    for idx, word in enumerate(all_words):
        if word in sentence_words:
            bag[idx] = 1.0

    return bag


# -----------------------------------------------------------
# Intent Data Management
# -----------------------------------------------------------
def load_intents(file_path):
    """Load intents from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            intents = json.load(file)
        return intents
    except FileNotFoundError:
        print(f"Error: Intents file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None


def preprocess_data(intents_data):
    """Preprocess the intents data for training."""
    words = []
    labels = []
    xy = []

    for intent in intents_data['intents']:
        tag = intent['tag']
        labels.append(tag)

        for pattern in intent['patterns']:
            # Tokenize each pattern
            w = tokenize(pattern)
            words.extend(w)
            xy.append((w, tag))

    # Remove duplicates and sort
    words = sorted(list(set(words)))
    labels = sorted(list(set(labels)))

    print(f"Found {len(words)} unique words")
    print(f"Found {len(labels)} unique labels")

    return words, labels, xy


def create_training_data(words, labels, xy):
    """Create training data from preprocessed data."""
    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, words)
        X_train.append(bag)
        y_train.append(labels.index(tag))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


# -----------------------------------------------------------
# Model Data Management
# -----------------------------------------------------------
def save_model_data(words, labels, model_path="models"):
    """Save model data (words and labels) to files."""
    os.makedirs(model_path, exist_ok=True)

    with open(f"{model_path}/words.pkl", 'wb') as f:
        pickle.dump(words, f)

    with open(f"{model_path}/labels.pkl", 'wb') as f:
        pickle.dump(labels, f)

    print(f"Model data saved to {model_path}/")


def load_model_data(model_path="models"):
    """Load model data (words and labels) from files."""
    try:
        with open(f"{model_path}/words.pkl", 'rb') as f:
            words = pickle.load(f)

        with open(f"{model_path}/labels.pkl", 'rb') as f:
            labels = pickle.load(f)

        print(f"Model data loaded from {model_path}/")
        return words, labels
    except FileNotFoundError:
        print(f"Error: Model data not found in {model_path}/")
        return None, None


# -----------------------------------------------------------
# Text Cleaning and Utilities
# -----------------------------------------------------------
def clean_text(text):
    """Clean and preprocess text input."""
    text = text.lower()
    text = ' '.join(text.split())
    return text


def get_response_patterns(intents_data, tag):
    """Get response patterns for a specific tag."""
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return intent['responses']
    return []


# -----------------------------------------------------------
# Testing
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Testing NLTK utilities...")

    test_sentence = "Hello! How are you doing today?"
    tokens = tokenize(test_sentence)
    print(f"Tokenized: {tokens}")

    test_words = ["running", "jumps", "better", "happily"]
    stemmed = [stem(word) for word in test_words]
    print(f"Stemmed: {stemmed}")

    all_words = ["hello", "how", "are", "you", "doing", "today"]
    bag = bag_of_words(tokens, all_words)
    print(f"Bag of words: {bag}")

    print("NLTK utilities test completed!")

