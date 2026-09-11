import time
from datetime import datetime, timezone
import logging
from app.models.schemas import (
    TextAnalysisRequest,
    ComprehensiveAnalysisResponse
)
from app.services.document_parser import DocumentParser
from app.services.finbert_service import FinBERTService
from app.services.bert_service import BERTRiskClassifierService
from app.services.genai_service import GenerativeAIService

logger = logging.getLogger("financial_api.pipeline")


class FinancialRiskPipeline:
    """Unified Orchestration Pipeline combining FinBERT, BERT, and Generative AI."""

    def __init__(
        self,
        finbert_service: FinBERTService,
        bert_service: BERTRiskClassifierService,
        genai_service: GenerativeAIService
    ):
        self.finbert = finbert_service
        self.bert = bert_service
        self.genai = genai_service

    async def run(self, request: TextAnalysisRequest) -> ComprehensiveAnalysisResponse:
        start_time = time.perf_counter()

        # Step 1: Document Parsing and Metadata Extraction
        metadata = DocumentParser.extract_metadata(request.text)

        # Step 2: FinBERT Tone & Sentiment Analysis
        finbert_res = self.finbert.analyze(request.text)

        # Step 3: BERT Multi-Category Risk Classification
        bert_res = self.bert.classify_risk(request.text)

        # Step 4: Generative AI Executive Synthesis (if requested)
        exec_summary = None
        if request.include_genai_summary:
            exec_summary = await self.genai.generate_executive_summary(
                text=request.text,
                doc_type=request.doc_type.value,
                finbert_result=finbert_res,
                bert_result=bert_res
            )

        # Step 5: Generative AI Mitigation Plan (if requested)
        mitigation_plan = None
        if request.include_mitigation:
            mitigation_plan = await self.genai.generate_mitigation_plan(
                text=request.text,
                company_name=request.company_name,
                bert_result=bert_res
            )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        now_utc = datetime.now(timezone.utc).isoformat()

        return ComprehensiveAnalysisResponse(
            document_metadata=metadata,
            finbert_sentiment=finbert_res,
            bert_risk_profile=bert_res,
            executive_summary=exec_summary,
            mitigation_plan=mitigation_plan,
            analysis_timestamp=now_utc,
            processing_time_ms=elapsed_ms
        )
