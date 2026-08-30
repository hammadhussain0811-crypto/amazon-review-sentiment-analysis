import logging
import yaml
from pathlib import Path

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score,
)

from src.common.preprocessing_util import preprocess_text
from src.training.config.settings import Settings

from nltk.corpus import stopwords


def train_model():

    try:
        # Load settings
        settings = Settings()

        DATASET_PATH = Path(settings.sentiment_dataset_path)
        MODEL_PATH = Path(settings.sentiment_model_path)
        LOG_PATH = Path(settings.log_path)
        HYPER_PARAMS_YAML_PATH = Path(settings.hyper_params_yaml_path)

        TEST_SIZE = settings.test_size
        RANDOM_STATE = settings.random_state

        # Create required directories
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH),
            ],
        )

        logging.info("Starting Sentiment Model Training")

        
        # 1. Loading dataset

        df = pd.read_csv(DATASET_PATH)

        logging.info(f"Dataset loaded with shape: {df.shape}")

        # 2. Creatimg combined text

        df["summary"] = df["summary"].fillna("")
        df["reviewText"] = df["reviewText"].fillna("")

        df["text"] = df["summary"] + " " + df["reviewText"]

        # 3. Converting rating into sentiment

        df["sentiment"] = df["rating"].apply(
            lambda rating: 0 if rating <= 3 else 1
        )

        # 4. Text preprocessing

        stop_words = set(stopwords.words("english"))

        # Keep important negation words
        stop_words = stop_words - {
            "not",
            "no",
            "nor",
            "never",
        }

        df["clean_text"] = df["text"].apply(
            lambda text: preprocess_text(text, stop_words)
        )

        # 5. Separate X and y

        X = df["clean_text"]
        y = df["sentiment"]

        # 6. Train/Test Split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        logging.info(
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        # 7. Load hyperparameters

        with open(HYPER_PARAMS_YAML_PATH, "r") as file:
            hyperparams = yaml.safe_load(file)

        model_params = hyperparams["sentiment"]["params"]

        
        # 8. Create TF-IDF + Logistic Regression pipeline

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        random_state=RANDOM_STATE,
                        **model_params,
                    ),
                ),
            ]
        )
        # 9. Training the model
        
        pipeline.fit(X_train, y_train)

        logging.info("Model training completed")

        
        # 10. Evaluation

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(
            f"Train Accuracy: {train_acc:.4f} | "
            f"Recall: {train_recall:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        logging.info(
            f"Test Accuracy: {test_acc:.4f} | "
            f"Recall: {test_recall:.4f} | "
            f"F1: {test_f1:.4f}"
        )

        logging.info(
            "Test Classification Report:\n"
            + classification_report(
                y_test,
                y_test_pred,
                target_names=["Negative", "Positive"],
            )
        )

        # 11. Saving the complete pipeline        

        dump(pipeline, MODEL_PATH)

        logging.info(f"Model saved to: {MODEL_PATH}")

        logging.info("Sentiment Training Completed")

    except Exception as e:

        logging.exception(f"Training failed: {e}")

        raise


if __name__ == "__main__":
    train_model()