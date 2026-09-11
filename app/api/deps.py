from functools import lru_cache
from fastapi import Depends
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService
from app.services.genai_service import GenerativeAIService
from app.services.risk_pipeline import FinancialRiskPipeline


@lru_cache()
def get_finbert_service() -> FinBERTService:
    """Returns singleton instance of FinBERT service."""
    return FinBERTService()


@lru_cache()
def get_bert_service() -> BERTRiskClassifierService:
    """Returns singleton instance of BERT Risk Classifier service."""
    return BERTRiskClassifierService()


@lru_cache()
def get_genai_service() -> GenerativeAIService:
    """Returns singleton instance of Generative AI service."""
    return GenerativeAIService()


def get_risk_pipeline(
    finbert: FinBERTService = Depends(get_finbert_service),
    bert: BERTRiskClassifierService = Depends(get_bert_service),
    genai: GenerativeAIService = Depends(get_genai_service)
) -> FinancialRiskPipeline:
    """Constructs the unified risk pipeline dependency."""
    return FinancialRiskPipeline(finbert, bert, genai)
