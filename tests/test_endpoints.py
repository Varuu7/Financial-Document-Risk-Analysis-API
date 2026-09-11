import io
import pytest

SAMPLE_FINANCIAL_TEXT = (
    "Due to macroeconomic deterioration and continuous supply chain delays, our gross profit "
    "declined by 18% year-over-year. The company faces ongoing liquidity constraints and may "
    "fail to satisfy debt covenant requirements under our revolving loan agreement. "
    "Furthermore, a pending SEC regulatory investigation could impose substantial monetary penalties."
)


def test_analyze_text_endpoint(client):
    payload = {
        "text": SAMPLE_FINANCIAL_TEXT,
        "doc_type": "10-K",
        "company_name": "Test Financial Corp",
        "include_genai_summary": True,
        "include_mitigation": True
    }
    response = client.post("/api/v1/analyze/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "document_metadata" in data
    assert "finbert_sentiment" in data
    assert "bert_risk_profile" in data
    assert "executive_summary" in data
    assert "mitigation_plan" in data
    assert data["processing_time_ms"] > 0


def test_analyze_file_endpoint(client):
    file_bytes = io.BytesIO(SAMPLE_FINANCIAL_TEXT.encode("utf-8"))
    files = {"file": ("filing_risk_factors.txt", file_bytes, "text/plain")}
    data = {
        "doc_type": "10-K",
        "company_name": "TestCorp",
        "include_genai_summary": "true",
        "include_mitigation": "true"
    }
    response = client.post("/api/v1/analyze/file", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["document_metadata"]["total_words"] > 0
    assert res_data["bert_risk_profile"]["overall_risk_score"] >= 0.0


def test_finbert_sentiment_endpoint(client):
    payload = {"text": "Operating revenues fell significantly and net loss widened."}
    response = client.post("/api/v1/finbert/sentiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dominant_sentiment"] in ("negative", "neutral", "positive")
    assert "probabilities" in data
    assert "sentiment_polarity_index" in data


def test_bert_risk_categories_endpoint(client):
    payload = {
        "text": "The company may default on its debt obligations due to severe liquidity shortages and covenant breaches."
    }
    response = client.post("/api/v1/bert/risk-categories", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["categories"]) == 5
    assert data["overall_risk_score"] > 0.0


def test_generative_summary_endpoint(client):
    payload = {
        "text": "Revenues fell 10% due to FX volatility, and a cyberattack disrupted warehouse fulfillment.",
        "doc_type": "earnings_call"
    }
    response = client.post("/api/v1/generative/summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "executive_summary" in data
    assert "primary_concern" in data
    assert len(data["key_risk_drivers"]) > 0


def test_generative_mitigation_endpoint(client):
    payload = {
        "text": "Severe supply chain dependency on single-source vendors in Asia creates delivery bottlenecks.",
        "company_name": "Acme Electronics"
    }
    response = client.post("/api/v1/generative/mitigation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["mitigation_actions"]) > 0
    assert len(data["governance_recommendations"]) > 0


def test_generative_qa_endpoint(client):
    payload = {
        "document_text": "The company has $200 million in debt maturing in October 2026.",
        "question": "When does the company's debt mature?"
    }
    response = client.post("/api/v1/generative/qa", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] > 0.0
    assert "October 2026" in data["answer"] or len(data["relevant_excerpts"]) > 0
