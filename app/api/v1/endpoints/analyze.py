from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import Optional
from app.models.schemas import (
    TextAnalysisRequest,
    ComprehensiveAnalysisResponse,
    DocumentType
)
from app.api.deps import get_risk_pipeline
from app.services.risk_pipeline import FinancialRiskPipeline
from app.services.document_parser import DocumentParser

router = APIRouter()


@router.post(
    "/text",
    response_model=ComprehensiveAnalysisResponse,
    summary="Comprehensive Risk Audit (Text)",
    description=(
        "Performs a complete 360-degree financial risk audit on raw text disclosures. "
        "Integrates FinBERT sentiment, BERT 5-pillar risk classification, and Generative AI "
        "executive risk summaries and mitigation roadmaps."
    )
)
async def analyze_text(
    request: TextAnalysisRequest,
    pipeline: FinancialRiskPipeline = Depends(get_risk_pipeline)
) -> ComprehensiveAnalysisResponse:
    try:
        return await pipeline.run(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk analysis processing failed: {str(e)}"
        )


@router.post(
    "/file",
    response_model=ComprehensiveAnalysisResponse,
    summary="Comprehensive Risk Audit (File Upload)",
    description=(
        "Upload a financial document (.txt, .json, .csv, .pdf) for comprehensive risk profiling. "
        "Extracts text, calculates document statistics, applies FinBERT and BERT, and synthesizes "
        "Generative AI executive insights."
    )
)
async def analyze_file(
    file: UploadFile = File(..., description="Financial document file to analyze"),
    doc_type: DocumentType = Form(default=DocumentType.GENERAL),
    company_name: Optional[str] = Form(default=None),
    include_genai_summary: bool = Form(default=True),
    include_mitigation: bool = Form(default=True),
    pipeline: FinancialRiskPipeline = Depends(get_risk_pipeline)
) -> ComprehensiveAnalysisResponse:
    try:
        content = await file.read()
        extracted_text = DocumentParser.parse_uploaded_file(file.filename, content)
        
        if len(extracted_text.strip()) < 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Extracted document content is too short for financial risk analysis (minimum 20 characters required)."
            )

        request = TextAnalysisRequest(
            text=extracted_text,
            doc_type=doc_type,
            company_name=company_name,
            include_genai_summary=include_genai_summary,
            include_mitigation=include_mitigation
        )
        return await pipeline.run(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {str(e)}"
        )
