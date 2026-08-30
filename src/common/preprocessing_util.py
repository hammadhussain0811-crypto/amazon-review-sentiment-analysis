import re

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemma = WordNetLemmatizer()


def preprocess_text(text, stop_words):
    """
    Clean and preprocess a review text.
    """

    # Remove HTML
    review = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    review = re.sub(r"http\S+|www\S+", " ", review)

    # Keep only alphabets
    review = re.sub(r"[^a-zA-Z]", " ", review)

    # Lowercase
    review = review.lower()

    # Tokenize
    review = word_tokenize(review)

    # Remove stopwords and lemmatize
    review = [
        lemma.lemmatize(word)
        for word in review
        if word not in stop_words
    ]

    # Join words
    review = " ".join(review)

    # Remove extra spaces
    review = " ".join(review.split())

    return review