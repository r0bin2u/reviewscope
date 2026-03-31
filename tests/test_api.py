Ciimport pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_classifier):
    """Create a FastAPI test client with mocked model dependency."""
    from src.api.main import app
    from src.api.dependencies import get_model

    app.dependency_overrides[get_model] = lambda: mock_classifier
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    def test_predict_positive_review(self, client, mock_classifier):
        mock_classifier.predict.return_value = ("positive", 0.9823)
        response = client.post("/predict", json={"text": "The food was amazing!"})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.9823
        assert data["text"] == "The food was amazing!"
        assert data["model_version"] == "v1.0"

    def test_predict_negative_review(self, client, mock_classifier):
        mock_classifier.predict.return_value = ("negative", 0.9512)
        response = client.post("/predict", json={"text": "Terrible service."})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "negative"
        assert data["confidence"] == 0.9512

    def test_predict_neutral_review(self, client, mock_classifier):
        mock_classifier.predict.return_value = ("neutral", 0.7100)
        response = client.post("/predict", json={"text": "It was okay."})
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] == "neutral"
        assert data["confidence"] == 0.71

    def test_predict_calls_model_with_correct_text(self, client, mock_classifier):
        client.post("/predict", json={"text": "Great place!"})
        mock_classifier.predict.assert_called_once_with("Great place!")

    def test_predict_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_text_field_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_content_type(self, client):
        response = client.post("/predict", content="not json")
        assert response.status_code == 422
