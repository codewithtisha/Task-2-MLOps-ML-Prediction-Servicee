
"""Flask API for the saved text-classification model."""

import json
import pickle
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "text_classifier.joblib"
METRICS_PATH = PROJECT_DIR / "model" / "metrics.json"

# Large enough for any valid 10,000-character text, even when fully
# JSON-escaped, while rejecting oversized bodies before they are parsed.
MAX_REQUEST_BYTES = 128 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

model = None
categories = []
model_version = "unknown"


class ModelLoadError(Exception):
    """Raised when the model artifacts are missing or invalid."""


def read_model():
    """Load the model file, reporting expected failures as ModelLoadError."""
    try:
        return joblib.load(MODEL_PATH)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
        ValueError,
        KeyError,
    ) as error:
        # These come from missing, corrupted, or incompatible model files.
        # joblib's pure-Python unpickler raises KeyError on unknown opcodes.
        raise ModelLoadError(f"Could not read model file: {error}") from error


def read_metadata():
    """Load and validate the model categories and version."""
    try:
        with METRICS_PATH.open(encoding="utf-8") as file:
            metrics = json.load(file)
    except (OSError, ValueError) as error:
        raise ModelLoadError(f"Could not read metrics file: {error}") from error

    if not isinstance(metrics, dict):
        raise ModelLoadError("Metrics file must contain a JSON object.")

    loaded_categories = metrics.get("categories")
    if (
        not isinstance(loaded_categories, list)
        or not loaded_categories
        or not all(isinstance(name, str) for name in loaded_categories)
    ):
        raise ModelLoadError("Model categories are missing or invalid.")

    loaded_version = metrics.get("model_version")
    if not isinstance(loaded_version, str) or not loaded_version.strip():
        raise ModelLoadError("Model version is missing or invalid.")

    return loaded_categories, loaded_version


def load_model():
    """Load the saved model and its metadata."""
    global model, categories, model_version

    # Start unavailable so a failed load never leaves stale model data behind.
    model = None
    categories = []
    model_version = "unknown"

    try:
        loaded_model = read_model()
        loaded_categories, loaded_version = read_metadata()
    except ModelLoadError as error:
        app.logger.error("Could not load model artifacts: %s", error)
        return

    model = loaded_model
    categories = loaded_categories
    model_version = loaded_version

    app.logger.info("Model version %s loaded successfully.", model_version)


load_model()


@app.errorhandler(413)
def request_too_large(error):
    """Return a JSON error when the request body exceeds the size limit."""
    return jsonify({
        "error": "Request body is too large."
    }), 413


@app.get("/health")
def health():
    """Report whether the model is available."""
    if model is None or not categories:
        return jsonify({
            "status": "unavailable",
            "model_loaded": False,
        }), 503

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_version": model_version,
    }), 200


@app.post("/predict")
def predict():
    """Predict the newsgroup category for submitted text."""
    if model is None or not categories:
        return jsonify({
            "error": "Model is unavailable. Check the model artifacts."
        }), 503

    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json."
        }), 415

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object."
        }), 400

    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "'text' must be a non-empty string."
        }), 400

    if len(text) > 10000:
        return jsonify({
            "error": "'text' must not exceed 10000 characters."
        }), 400

    try:
        predicted_class = int(model.predict([text])[0])

        if not 0 <= predicted_class < len(categories):
            raise ValueError("Prediction index is outside the category list.")

        return jsonify({
            "prediction": categories[predicted_class],
            "model_version": model_version,
        }), 200

    except (ValueError, TypeError, IndexError):
        app.logger.exception("Prediction failed.")
        return jsonify({
            "error": "The model could not produce a valid prediction."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
