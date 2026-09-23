import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.app import app  # noqa: E402


client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_endpoint_returns_translation_payload():
    response = client.post("/translate", json={"text": "hello"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["translation"]
    assert payload["source"] in {"dataset_exact", "ai_guess"}
    assert isinstance(payload["is_exact_dataset_match"], bool)
