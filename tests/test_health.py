def test_root_redirect(client):
    """Test that root endpoint redirects to Swagger UI."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_swagger_docs_available(client):
    """Test that Swagger UI HTML is served at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_openapi_json_available(client):
    """Test that OpenAPI schema is valid and contains expected tags."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert data["info"]["title"] == "Financial Document Risk Analysis API"
    assert "/api/v1/analyze/text" in data["paths"]
    assert "/api/v1/finbert/sentiment" in data["paths"]
    assert "/api/v1/bert/risk-categories" in data["paths"]


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "device" in data
    assert "finbert_status" in data
    assert "bert_risk_status" in data
    assert "genai_engine" in data
