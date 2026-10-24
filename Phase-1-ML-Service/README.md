# Task 2 — Phase 1: ML Prediction Service

## Overview

This project trains a text-classification model and serves predictions through a Flask HTTP API. It demonstrates a reproducible machine-learning workflow, model evaluation, automated testing, input validation, and Docker containerization.

## Dataset

- **Name:** 20 Newsgroups
- **Source:** scikit-learn's `fetch_20newsgroups` dataset loader.
- **Official documentation:** https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
- **Task:** Classify text into one of four categories: `comp.graphics`, `rec.sport.baseball`, `sci.space`, and `talk.politics.misc`.

The training script downloads the dataset when necessary and uses the local cache when available. It removes headers, footers, and quoted text where possible.

**Data limitations:** The dataset contains historical posts. Filtering is heuristic and may not remove all identifying details. Review the dataset's terms and applicable privacy and redistribution considerations before sharing data or excerpts. Historical text may not represent modern language.

## Model and Training Workflow

- **Pipeline:** TF-IDF vectorization followed by Logistic Regression.
- **Model version:** `1.0.0`
- **Random seed:** `42`
- **Data split:** 80% training and 20% testing, stratified by class.
- **Evaluation metrics:** Accuracy, precision, recall, and F1-score.
- **Training script:** `train.py`
- **API application:** `app.py`
- **Saved model:** `model/text_classifier.joblib`
- **Saved metrics:** `model/metrics.json`

The training script loads the dataset, creates a reproducible split, trains and evaluates the model, and saves the model artifact and evaluation metrics. The Flask application loads the saved model and does not retrain it when handling requests.

## Separation of Training, Model Artifact, and Serving API

The project separates model training, model storage, and prediction serving into independent responsibilities. This makes the workflow easier to maintain, test, and reproduce.

## 1. Training Code — `train.py`

The training script is responsible for loading the 20 Newsgroups dataset, preparing the data, creating a reproducible train-test split, training the TF-IDF and Logistic Regression pipeline, and evaluating its performance using accuracy, precision, recall, and F1-score.

After training, it saves the trained pipeline to `model/text_classifier.joblib` and writes the evaluation results and model metadata to `model/metrics.json`.

Training is performed separately by running:

```powershell
uv run python train.py
```

## 2. Model Artifact — `model/text_classifier.joblib`

The model artifact is the saved, trained machine-learning pipeline. It contains the fitted TF-IDF vectorizer and Logistic Regression classifier needed to transform input text and generate predictions.

Keeping the trained model in a separate file allows it to be loaded and reused without retraining the model for every prediction request. The associated `model/metrics.json` file stores evaluation results and metadata, including the model version and supported categories.

## 3. Serving API — `app.py`

The Flask application is responsible for serving predictions over HTTP. When the application starts, it loads the saved model and its metadata. It provides `GET /health` to report model availability and `POST /predict` to accept text and return a predicted category.

The API validates incoming requests and returns appropriate HTTP error responses for invalid input or unavailable model artifacts. It does not perform model training when handling prediction requests.

## How the Components Work Together

The workflow follows three stages:

1. **Train:** `train.py` trains and evaluates the machine-learning pipeline.
2. **Save and load:** The trained pipeline is saved as `model/text_classifier.joblib`, with metrics and metadata in `model/metrics.json`. The API loads these files when it starts.
3. **Serve predictions:** A client sends a request to `POST /predict`. The API validates the text, uses the loaded model to predict its category, and returns the result as JSON.

This separation means that model training can be performed independently of the API, while the serving application can reuse the saved model for multiple prediction requests without retraining it each time.

## Evaluation Results

Results from the initial training run:

| Metric | Result |
|---|---:|
| Training examples | 2,983 |
| Test examples | 746 |
| Accuracy | 88.07% |

These results establish a baseline and do not guarantee performance on unseen real-world text. Results may vary with changes to the data, dependencies, or training configuration.

## API Endpoints

The Flask application listens on port `5001` inside the local development environment and Docker container.

### `GET /health`

Checks whether the model is available.

Successful response: HTTP `200 OK`.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "1.0.0"
}
```

If the model is unavailable, the endpoint returns HTTP `503 Service Unavailable`:

```json
{
  "status": "unavailable",
  "model_loaded": false
}
```

### `POST /predict`

Accepts a JSON object with a `text` field containing a non-empty string of at most 10,000 characters. The request must use `Content-Type: application/json`, and the request body must not exceed 128 KB.

Example request:

```json
{
  "text": "NASA launched a spacecraft into orbit"
}
```

Example response:

```json
{
  "prediction": "sci.space",
  "model_version": "1.0.0"
}
```

### Error Responses

All `/predict` errors return a JSON object with an `error` message, for example:

```json
{
  "error": "'text' must be a non-empty string."
}
```

| Status | When it is returned |
|---|---|
| `400 Bad Request` | Malformed JSON; a JSON body that is not an object; `text` missing, not a string, empty, or whitespace-only; `text` longer than 10,000 characters. |
| `413 Payload Too Large` | Request body larger than 128 KB. Returns `{"error": "Request body is too large."}`. |
| `415 Unsupported Media Type` | Request does not use the JSON content type. |
| `500 Internal Server Error` | The model returned a prediction that does not match a known category. |
| `503 Service Unavailable` | The model or its metadata could not be loaded. |

The checks run in this order: model availability (`503`), content type (`415`), body size (`413`), then input validation (`400`).

The 128 KB body limit is set through Flask's `MAX_CONTENT_LENGTH`. It is large enough for any valid 10,000-character text, even when every character is JSON-escaped, and stops oversized requests before they are parsed.

### Model Loading

The model and `model/metrics.json` are loaded once when the application starts. If an expected loading problem occurs, such as a missing, corrupted, or incompatible model file, or missing or invalid categories or model version in the metrics file, the error is logged and the service stays running in an unavailable state: `/health` and `/predict` return `503`. Unexpected errors are not suppressed and stop the application from starting.

## Setup and Reproducibility

**Python version:** 3.12.9

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

Install and synchronize the project environment:

```powershell
uv sync
```

Train the model:

```powershell
uv run python train.py
```

Run automated tests (either command works; `pyproject.toml` adds the project root to pytest's import path):

```powershell
uv run pytest -v
uv run python -m pytest -v
```

Start the API locally:

```powershell
uv run python app.py
```

The local API is available at `http://127.0.0.1:5001`.

The first training run requires internet access unless the dataset is already cached.

## Testing and Validation

### Verified Automated Test Results

In the latest verified run, all **24 test cases passed** with both `uv run pytest -v` and `uv run python -m pytest -v`. The tests are in `tests/test_app.py` (17 test functions, some parametrized over several inputs). They use Flask's test client and the saved model. Failure tests use pytest's temporary directory and monkeypatching, so the real model artifacts are never modified.

| Area | What the tests check |
|---|---|
| Health | `200` when the model is loaded; `503` with `model_loaded: false` when it is not. |
| Prediction | A space-related sentence is classified as `sci.space`; text of exactly 10,000 characters is accepted. |
| Input validation (`400`) | Empty text, whitespace-only text, missing `text`, non-string `text` (number, `null`, list, object, boolean), text over 10,000 characters, malformed JSON, and JSON bodies that are not objects (array, string, number, `null`). |
| Content type (`415`) | A plain-text request is rejected. |
| Body size (`413`) | A body over 128 KB returns `413` with a JSON `error`. |
| Model unavailable (`503`) | `/predict` returns `503` when no model is loaded. |
| Model loading | A missing model file and a corrupted model file leave the service unavailable with no stale model data; an unexpected error during loading is raised instead of being suppressed. |

### Not Covered by the Latest Verification

The following have **not** been verified since the current changes were made:

- **Docker:** The image was built and the container was tested through the host port mapping before the latest changes to `app.py`. The image has not been rebuilt or retested since.
- **Manual API testing:** Health checks, predictions, and invalid input were tested manually before the latest changes. They have not been manually retested since; the current behavior is covered by the automated tests above.
- **Untested code paths:** the `500` response for an out-of-range prediction, invalid content in `metrics.json` (wrong types or missing fields), other kinds of corrupted model files, and the body limit for streamed requests with no declared size.
- **Other routes:** Requests to unknown routes (`404`) or with an unsupported method (`405`) return Flask's default HTML responses, not JSON.

### Manual Checks

To check the health endpoint locally:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5001/health -Method Get
```

To test a prediction locally:

```powershell
$body = @{ text = "NASA launched a spacecraft into orbit" } | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:5001/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Docker

Build the Docker image:

```powershell
docker build -t task2-ml-service:1.0 .
```

Run the container, mapping host port `5002` to container port `5001`:

```powershell
docker run -d --name task2-ml-container -p 5002:5001 task2-ml-service:1.0
```

Check the container health endpoint:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5002/health -Method Get
```

The containerized API is available at `http://127.0.0.1:5002`.

If the named container is already running, reuse it rather than creating a duplicate.

**Note:** The Docker workflow was validated before the latest changes to `app.py`. Rebuild the image to include the current code. The rebuilt image has not been verified yet.

The container runs the application with Flask's built-in server, which is intended for development and demonstration rather than production use.

## Project Structure

```text
Task-2-Phase-1-ML-Service/
├── app.py
├── train.py
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── README.md
├── model/
│   ├── text_classifier.joblib
│   └── metrics.json
└── tests/
    └── test_app.py
```

## Project Status

The initial ML model has been trained and evaluated, and the saved model artifact and metrics are available. The Flask API includes input validation, a 128 KB request-body limit with a JSON `413` response, and safer model loading. All 24 automated test cases passed in the latest verified run. Docker build and container API validation were completed before the latest `app.py` changes and still need to be repeated.

**Remaining submission activities:** Final documentation review, AI-assisted workflow log, screenshots, test evidence, and narrated demonstration recording.