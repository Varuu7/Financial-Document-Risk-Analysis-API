from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from app.models.schemas import BERTRiskResponse
from app.api.deps import get_bert_service
from app.services.bert_service import BERTRiskClassifierService

router = APIRouter()


class BERTRiskInput(BaseModel):
    text: str = Field(
        ...,
        min_length=15,
        description="Financial disclosures or Item 1A Risk Factors excerpt",
        examples=[
            "Our revolving credit facility matures in June 2027 and contains strict leverage covenants. "
            "If counterparty banks fail to renew our commitments or if revenues decline further, "
            "we may experience acute liquidity shortages."
        ]
    )


@router.post(
    "/risk-categories",
    response_model=BERTRiskResponse,
    summary="BERT Multi-Category Risk Classification",
    description=(
        "Classifies disclosures across 5 standardized financial risk pillars: Credit Risk, Market Risk, "
        "Liquidity Risk, Operational Risk, and Regulatory & Legal Risk. Returns severity scores (0.0 to 1.0), "
        "risk levels (LOW, MEDIUM, HIGH, CRITICAL), and extracts top high-risk sentence clauses."
    )
)
async def classify_bert_risks(
    payload: BERTRiskInput,
    bert: BERTRiskClassifierService = Depends(get_bert_service)
) -> BERTRiskResponse:
    try:
        return bert.classify_risk(payload.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BERT risk classification failed: {str(e)}"
        )
