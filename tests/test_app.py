from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'status' in resp.json()
