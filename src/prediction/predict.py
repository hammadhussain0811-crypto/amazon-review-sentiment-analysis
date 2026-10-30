
from pathlib import Path

from joblib import load
from nltk.corpus import stopwords

from src.common.preprocessing_util import preprocess_text


# 1. Get project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# 2. Model path
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.pkl"


# 3. Loading the trained pipeline
pipeline = load(MODEL_PATH)


# 4. Loading stopwords
stop_words = set(stopwords.words("english"))

# Preserving the important negation words
stop_words = stop_words - {"not", "no", "nor", "never"}


# 5. Prediction function
def predict_sentiment(review):

    # Preprocess new review
    clean_review = preprocess_text(review, stop_words)

    # Predict using saved pipeline
    prediction = pipeline.predict([clean_review])[0]

    if prediction == 1:
        return "Positive"

    return "Negative"


# 6. Test prediction
if __name__ == "__main__":

    review = "This book is absolutely amazing. I loved reading it!"

    result = predict_sentiment(review)

    print("Review:", review)
    print("Predicted Sentiment:", result)
