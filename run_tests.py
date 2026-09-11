import unittest
import io
import json
from fastapi.testclient import TestClient
from app.main import app
from app.services.document_parser import DocumentParser
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService
from app.services.genai_service import GenerativeAIService
from app.models.schemas import RiskLevel


class TestFinancialRiskAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_text = (
            "Due to macroeconomic deterioration and continuous supply chain delays, our gross profit "
            "declined by 18% year-over-year. The company faces ongoing liquidity constraints and may "
            "fail to satisfy debt covenant requirements under our revolving loan agreement. "
            "Furthermore, a pending SEC regulatory investigation could impose substantial monetary penalties."
        )

    def test_01_root_redirects_to_swagger(self):
        res = self.client.get("/", follow_redirects=False)
        self.assertIn(res.status_code, (302, 307))
        self.assertEqual(res.headers["location"], "/docs")

    def test_02_swagger_docs_html(self):
        res = self.client.get("/docs")
        self.assertEqual(res.status_code, 200)
        self.assertIn("swagger-ui", res.text.lower())

    def test_03_openapi_schema(self):
        res = self.client.get("/openapi.json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("openapi", data)
        self.assertEqual(data["info"]["title"], "Financial Document Risk Analysis API")
        self.assertIn("/api/v1/analyze/text", data["paths"])
        self.assertIn("/api/v1/analyze/file", data["paths"])
        self.assertIn("/api/v1/finbert/sentiment", data["paths"])
        self.assertIn("/api/v1/bert/risk-categories", data["paths"])
        self.assertIn("/api/v1/generative/summary", data["paths"])
        self.assertIn("/api/v1/generative/mitigation", data["paths"])
        self.assertIn("/api/v1/generative/qa", data["paths"])

    def test_04_health_endpoint(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")

    def test_05_finbert_sentiment_endpoint(self):
        res = self.client.post(
            "/api/v1/finbert/sentiment",
            json={"text": "Operating revenues contracted severely with catastrophic asset impairments."}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["dominant_sentiment"], "negative")
        self.assertLess(data["sentiment_polarity_index"], 0.0)
        self.assertIn("probabilities", data)

    def test_06_bert_risk_categories_endpoint(self):
        res = self.client.post(
            "/api/v1/bert/risk-categories",
            json={"text": "The borrower has breached credit covenants and is at risk of bankruptcy liquidation."}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["categories"]), 5)
        self.assertGreater(data["overall_risk_score"], 0.0)
        credit_risk = next(c for c in data["categories"] if c["category"] == "Credit Risk")
        self.assertIn(credit_risk["risk_level"], ["HIGH", "CRITICAL"])

    def test_07_generative_summary_endpoint(self):
        res = self.client.post(
            "/api/v1/generative/summary",
            json={"text": self.sample_text, "doc_type": "10-K"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(len(data["executive_summary"]) > 20)
        self.assertTrue(len(data["key_risk_drivers"]) > 0)

    def test_08_generative_mitigation_endpoint(self):
        res = self.client.post(
            "/api/v1/generative/mitigation",
            json={"text": self.sample_text, "company_name": "Acme Holdings"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(len(data["mitigation_actions"]) > 0)
        self.assertTrue(len(data["governance_recommendations"]) > 0)

    def test_09_generative_qa_endpoint(self):
        res = self.client.post(
            "/api/v1/generative/qa",
            json={
                "document_text": "The company has $2.1 billion in outstanding debt maturing in November 2026.",
                "question": "What is the total outstanding debt and when does it mature?"
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["confidence"], 0.0)
        self.assertTrue(len(data["answer"]) > 10)

    def test_10_comprehensive_text_analysis(self):
        res = self.client.post(
            "/api/v1/analyze/text",
            json={
                "text": self.sample_text,
                "doc_type": "10-K",
                "company_name": "Apex Global Inc.",
                "include_genai_summary": True,
                "include_mitigation": True
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("document_metadata", data)
        self.assertIn("finbert_sentiment", data)
        self.assertIn("bert_risk_profile", data)
        self.assertIn("executive_summary", data)
        self.assertIn("mitigation_plan", data)
        self.assertGreater(data["processing_time_ms"], 0.0)

    def test_11_comprehensive_file_upload(self):
        file_bytes = io.BytesIO(self.sample_text.encode("utf-8"))
        files = {"file": ("filing.txt", file_bytes, "text/plain")}
        data = {
            "doc_type": "10-K",
            "company_name": "Apex Global Inc.",
            "include_genai_summary": "true",
            "include_mitigation": "true"
        }
        res = self.client.post("/api/v1/analyze/file", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        res_data = res.json()
        self.assertGreater(res_data["document_metadata"]["total_words"], 20)
        self.assertIn("bert_risk_profile", res_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
