import pytest
from app.services.document_parser import DocumentParser
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService
from app.services.genai_service import GenerativeAIService
from app.models.schemas import RiskLevel


def test_document_parser():
    sample_text = (
        "Operating profit dropped 15% due to rising fuel costs. "
        "U.S. inflation remains elevated. Counterparty risk has increased."
    )
    metadata = DocumentParser.extract_metadata(sample_text)
    assert metadata.total_words > 0
    assert metadata.total_sentences >= 2
    assert metadata.estimated_reading_time_mins > 0

    chunks = DocumentParser.chunk_text(sample_text, chunk_size=10, overlap=3)
    assert len(chunks) >= 1


def test_finbert_service():
    finbert = FinBERTService()
    text = "We suffered a devastating financial loss and experienced severe credit rating downgrades."
    res = finbert.analyze(text)
    assert res.dominant_sentiment in ("negative", "neutral", "positive")
    assert -1.0 <= res.sentiment_polarity_index <= 1.0
    assert 0.0 <= res.probabilities.negative <= 1.0
    assert 0.0 <= res.financial_uncertainty_score <= 1.0
    assert len(res.interpretation) > 10


def test_bert_risk_classifier():
    bert = BERTRiskClassifierService()
    text = (
        "We are currently facing severe liquidity shortages and potential debt covenant defaults. "
        "Additionally, a ransomware cyberattack compromised key internal customer databases."
    )
    res = bert.classify_risk(text)
    assert 0.0 <= res.overall_risk_score <= 1.0
    assert res.overall_risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert len(res.categories) == 5
    
    # Check that liquidity risk or operational risk was detected
    cat_names = [c.category for c in res.categories]
    assert "Liquidity Risk" in cat_names
    assert "Operational Risk" in cat_names


@pytest.mark.asyncio
async def test_genai_service():
    genai = GenerativeAIService()
    text = "Operating margins declined due to supply chain bottlenecks and rising interest rates."
    
    summary = await genai.generate_executive_summary(text)
    assert summary.executive_summary is not None
    assert len(summary.key_risk_drivers) > 0
    
    mitigation = await genai.generate_mitigation_plan(text)
    assert len(mitigation.mitigation_actions) > 0
    assert len(mitigation.governance_recommendations) > 0

    qa = await genai.answer_question(text, "Why did operating margins decline?")
    assert qa.answer is not None
    assert qa.confidence > 0.0
