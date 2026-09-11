from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from app.models.schemas import (
    ExecutiveRiskSummary,
    MitigationResponse,
    RiskQARequest,
    RiskQAResponse,
    DocumentType
)
from app.api.deps import get_genai_service, get_finbert_service, get_bert_service
from app.services.genai_service import GenerativeAIService
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService

router = APIRouter()


class SummaryInput(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="Financial document text to summarize",
        examples=[
            "Revenue fell 12% year-over-year due to foreign exchange headwinds and severe supply chain "
            "delays in microchip components. In addition, antitrust scrutiny by the Department of Justice "
            "creates uncertainty around our planned enterprise cloud acquisition."
        ]
    )
    doc_type: DocumentType = Field(default=DocumentType.GENERAL)


class MitigationInput(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="Financial text containing identified risks",
        examples=[
            "Due to tightening credit markets and escalating debt service obligations, "
            "our working capital has diminished by 28%. We face potential covenant breaches "
            "if Q4 earnings fail to rebound."
        ]
    )
    company_name: Optional[str] = Field(default=None, examples=["Global Logistics Inc."])


@router.post(
    "/summary",
    response_model=ExecutiveRiskSummary,
    summary="Executive Risk Briefing (Generative AI)",
    description=(
        "Uses Generative AI to synthesize complex financial disclosures into a high-level executive "
        "risk summary, identifying the single primary concern, top risk drivers, and stabilizing factors."
    )
)
async def generate_executive_summary(
    payload: SummaryInput,
    genai: GenerativeAIService = Depends(get_genai_service),
    finbert: FinBERTService = Depends(get_finbert_service),
    bert: BERTRiskClassifierService = Depends(get_bert_service)
) -> ExecutiveRiskSummary:
    try:
        # Precompute quant signals to enrich the GenAI briefing
        finbert_res = finbert.analyze(payload.text)
        bert_res = bert.classify_risk(payload.text)
        return await genai.generate_executive_summary(
            text=payload.text,
            doc_type=payload.doc_type.value,
            finbert_result=finbert_res,
            bert_result=bert_res
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generative summary failed: {str(e)}"
        )


@router.post(
    "/mitigation",
    response_model=MitigationResponse,
    summary="Risk Mitigation Roadmap (Generative AI)",
    description=(
        "Generates structured, time-phased mitigation actions (30-day immediate, 90-day medium-term, "
        "strategic horizon) and enterprise governance recommendations tailored to the document's risks."
    )
)
async def generate_mitigation_plan(
    payload: MitigationInput,
    genai: GenerativeAIService = Depends(get_genai_service),
    bert: BERTRiskClassifierService = Depends(get_bert_service)
) -> MitigationResponse:
    try:
        bert_res = bert.classify_risk(payload.text)
        return await genai.generate_mitigation_plan(
            text=payload.text,
            company_name=payload.company_name,
            bert_result=bert_res
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mitigation plan generation failed: {str(e)}"
        )


@router.post(
    "/qa",
    response_model=RiskQAResponse,
    summary="Financial Risk Q&A Assistant (Generative AI)",
    description=(
        "Ask targeted questions regarding financial risks, debt covenants, regulatory threats, "
        "or operational exposures in the document. Answers are strictly grounded in document evidence."
    )
)
async def ask_financial_question(
    payload: RiskQARequest,
    genai: GenerativeAIService = Depends(get_genai_service)
) -> RiskQAResponse:
    try:
        return await genai.answer_question(
            document_text=payload.document_text,
            question=payload.question
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk Q&A failed: {str(e)}"
        )
