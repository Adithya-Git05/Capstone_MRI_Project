from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200

def test_case_schema():
    response = client.get('/api/cases/DEMO_001')
    assert response.status_code == 200
    assert response.json()['data_type'] == 'SYNTHETIC_DEMO'

def test_unknown_case():
    assert client.get('/api/cases/UNKNOWN').status_code == 404

def test_query_response_shape():
    response = client.post('/api/rag/query', json={'question': 'What is considered in glioma planning?', 'case_id': 'DEMO_001'})
    assert response.status_code == 200
    payload = response.json()
    assert 'verification' in payload and 'numerical_checks' in payload
    assert 'retrieval_diagnostics' in payload
    assert all('passage' in item for item in payload['retrieved_evidence'])
