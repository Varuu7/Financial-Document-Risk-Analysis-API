from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.models.schemas import FinBERTResponse
from app.api.deps import get_finbert_service
from app.services.finbert_service import FinBERTService

router = APIRouter()


class FinBERTInput(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Financial statement or disclosure excerpt to evaluate with FinBERT",
        examples=[
            "Operating income decreased by $45 million compared to the prior year period, "
            "driven by higher raw material costs and lower customer order volumes in North America."
        ]
    )


@router.post(
    "/sentiment",
    response_model=FinBERTResponse,
    summary="FinBERT Financial Sentiment & Polarity",
    description=(
        "Evaluates financial text using FinBERT (BERT fine-tuned on Financial PhraseBank). "
        "Returns discrete probabilities (Positive, Neutral, Negative), a net Sentiment Polarity "
        "Index (-1.0 to +1.0), and a financial uncertainty score."
    )
)
async def analyze_finbert_sentiment(
    payload: FinBERTInput,
    finbert: FinBERTService = Depends(get_finbert_service)
) -> FinBERTResponse:
    try:
        return finbert.analyze(payload.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FinBERT sentiment analysis failed: {str(e)}"
        )
