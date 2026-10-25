
"""Automated tests for the Flask prediction API."""

import pytest

import app as ml_app


def test_health_returns_200():
    client = ml_app.app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert response.get_json()["model_loaded"] is True


def test_valid_prediction():
    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        json={"text": "NASA launched a spacecraft into orbit."},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] == "sci.space"
    assert data["model_version"] == "1.1.0"


def test_empty_text_is_rejected():
    client = ml_app.app.test_client()
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_missing_text_is_rejected():
    client = ml_app.app.test_client()
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_non_json_request_is_rejected():
    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        data="plain text",
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert "error" in response.get_json()


def test_unavailable_model_returns_503(monkeypatch):
    client = ml_app.app.test_client()
    monkeypatch.setattr(ml_app, "model", None)

    response = client.post(
        "/predict",
        json={"text": "A spacecraft reached orbit."},
    )

    assert response.status_code == 503
    assert "error" in response.get_json()


@pytest.fixture
def restore_model_state(monkeypatch):
    """Let a test reload the model and restore the real state afterwards."""
    for name in ("model", "categories", "model_version", "MODEL_PATH"):
        monkeypatch.setattr(ml_app, name, getattr(ml_app, name))


def test_whitespace_only_text_is_rejected():
    client = ml_app.app.test_client()
    response = client.post("/predict", json={"text": "   \n\t"})

    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize("text", [123, None, ["a"], {"a": 1}, True])
def test_non_string_text_is_rejected(text):
    client = ml_app.app.test_client()
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_text_over_limit_is_rejected():
    client = ml_app.app.test_client()
    response = client.post("/predict", json={"text": "a" * 10001})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_text_at_limit_is_accepted():
    client = ml_app.app.test_client()
    response = client.post("/predict", json={"text": "a" * 10000})

    assert response.status_code == 200
    assert response.get_json()["prediction"] in ml_app.categories


def test_malformed_json_is_rejected():
    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        data='{"text": ',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize("body", ["[]", '"hello"', "42", "null"])
def test_non_object_json_is_rejected(body):
    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        data=body,
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_oversized_body_returns_413():
    client = ml_app.app.test_client()
    response = client.post(
        "/predict",
        json={"text": "a" * ml_app.MAX_REQUEST_BYTES},
    )

    assert response.status_code == 413
    assert response.is_json
    assert "error" in response.get_json()


def test_health_returns_503_when_model_unavailable(monkeypatch):
    client = ml_app.app.test_client()
    monkeypatch.setattr(ml_app, "model", None)

    response = client.get("/health")

    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "unavailable"
    assert data["model_loaded"] is False


def test_missing_model_file_leaves_service_unavailable(
    restore_model_state, monkeypatch, tmp_path
):
    monkeypatch.setattr(ml_app, "MODEL_PATH", tmp_path / "missing.joblib")

    ml_app.load_model()

    assert ml_app.model is None
    assert ml_app.categories == []
    assert ml_app.model_version == "unknown"
    response = ml_app.app.test_client().get("/health")
    assert response.status_code == 503


def test_corrupted_model_file_leaves_service_unavailable(
    restore_model_state, monkeypatch, tmp_path
):
    corrupted_path = tmp_path / "corrupted.joblib"
    corrupted_path.write_bytes(b"not a valid joblib file")
    monkeypatch.setattr(ml_app, "MODEL_PATH", corrupted_path)

    ml_app.load_model()

    assert ml_app.model is None
    assert ml_app.categories == []
    assert ml_app.model_version == "unknown"
    response = ml_app.app.test_client().get("/health")
    assert response.status_code == 503


def test_unexpected_loading_error_is_not_swallowed(
    restore_model_state, monkeypatch
):
    def broken_load(path):
        raise RuntimeError("unexpected programming error")

    monkeypatch.setattr(ml_app.joblib, "load", broken_load)

    with pytest.raises(RuntimeError):
        ml_app.load_model()

    # The service is still left unavailable rather than holding stale data.
    assert ml_app.model is None
    assert ml_app.categories == []
