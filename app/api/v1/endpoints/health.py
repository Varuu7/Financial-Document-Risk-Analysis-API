import torch
from fastapi import APIRouter, Depends
from app.config import get_settings, Settings
from app.models.schemas import HealthResponse
from app.api.deps import get_finbert_service, get_bert_service, get_genai_service
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService
from app.services.genai_service import GenerativeAIService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Model Readiness",
    description="Returns API status, hardware execution environment (CPU/CUDA), and engine readiness."
)
async def health_check(
    settings: Settings = Depends(get_settings),
    finbert: FinBERTService = Depends(get_finbert_service),
    bert: BERTRiskClassifierService = Depends(get_bert_service),
    genai: GenerativeAIService = Depends(get_genai_service)
):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        device_name += f" ({torch.cuda.get_device_name(0)})"

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        device=device_name,
        finbert_status=finbert.get_status(),
        bert_risk_status=bert.get_status(),
        genai_engine=genai.get_status()
    )
