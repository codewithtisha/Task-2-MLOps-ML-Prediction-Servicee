
"""Train and evaluate a reproducible text-classification model."""

import json
from pathlib import Path

import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Keep project paths independent of the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
MODEL_PATH = MODEL_DIR / "text_classifier.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Reproducibility settings.
RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"

CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc",
]


def main():
    # Download the public dataset, or use its local cache.
    dataset = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    # Separate training and test data using a fixed seed.
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=dataset.target,
    )

    # Bundle text processing and classification into one pipeline.
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            ),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate only on the held-out test set.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
        output_dict=True,
        zero_division=0,
    )

    # Save the trained model and evaluation results.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model_version": MODEL_VERSION,
        "random_seed": RANDOM_SEED,
        "dataset": "scikit-learn 20 Newsgroups",
        "categories": dataset.target_names,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": accuracy,
        "classification_report": report,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model version: {MODEL_VERSION}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
