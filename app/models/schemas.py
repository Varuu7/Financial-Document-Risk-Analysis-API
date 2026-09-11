from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DocumentType(str, Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    EARNINGS_CALL = "earnings_call"
    AUDIT_REPORT = "audit_report"
    CREDIT_AGREEMENT = "credit_agreement"
    PRESS_RELEASE = "press_release"
    GENERAL = "general"


# ------------------------------------------------------------------------------
# Document Input & Metadata
# ------------------------------------------------------------------------------
class TextAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="The financial text to analyze (e.g. 10-K risk factors, earnings call transcript, press release)",
        examples=[
            "Due to continuing supply chain disruptions and elevated foreign exchange volatility, "
            "operating margins declined by 320 basis points. Furthermore, ongoing antitrust litigation "
            "in Europe may result in material penalties and restrictions on our enterprise distribution."
        ]
    )
    doc_type: DocumentType = Field(
        default=DocumentType.GENERAL,
        description="Type of financial document"
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Company name or ticker symbol for contextual analysis",
        examples=["Acme Corp (ACM)"]
    )
    include_genai_summary: bool = Field(
        default=True,
        description="Whether to generate an executive AI risk summary"
    )
    include_mitigation: bool = Field(
        default=True,
        description="Whether to generate actionable AI risk mitigation strategies"
    )


class DocumentMetadata(BaseModel):
    total_characters: int
    total_words: int
    total_sentences: int
    estimated_reading_time_mins: float


# ------------------------------------------------------------------------------
# FinBERT Sentiment & Tone Schemas
# ------------------------------------------------------------------------------
class FinBERTProbabilities(BaseModel):
    positive: float = Field(..., ge=0.0, le=1.0, description="Probability of positive financial sentiment")
    neutral: float = Field(..., ge=0.0, le=1.0, description="Probability of neutral financial sentiment")
    negative: float = Field(..., ge=0.0, le=1.0, description="Probability of negative financial sentiment / distress")


class FinBERTResponse(BaseModel):
    dominant_sentiment: str = Field(..., description="Dominant sentiment label (positive, neutral, negative)")
    sentiment_polarity_index: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Net polarity score ranging from -1.0 (extreme distress/negative) to +1.0 (strong positive)"
    )
    probabilities: FinBERTProbabilities
    financial_uncertainty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Metric indicating degree of hedging, ambiguity, or uncertainty language"
    )
    interpretation: str = Field(..., description="Analyst-friendly interpretation of the FinBERT output")


# ------------------------------------------------------------------------------
# BERT Multi-Category Risk Classification Schemas
# ------------------------------------------------------------------------------
class CategoryRiskScore(BaseModel):
    category: str = Field(..., description="Financial risk category (e.g., Credit, Market, Liquidity, Operational, Regulatory)")
    score: float = Field(..., ge=0.0, le=1.0, description="Risk intensity score (0.0=minimal, 1.0=severe)")
    risk_level: RiskLevel = Field(..., description="Standardized severity level")
    key_factors: List[str] = Field(default_factory=list, description="Keywords and risk factors detected")
    sample_clauses: List[str] = Field(default_factory=list, description="Representative excerpt clauses")


class SentenceRiskAnnotation(BaseModel):
    sentence: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    primary_category: str
    clause_index: int


class BERTRiskResponse(BaseModel):
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Aggregated risk score across all categories")
    overall_risk_level: RiskLevel = Field(..., description="Global risk posture rating")
    categories: List[CategoryRiskScore] = Field(..., description="Risk evaluation across standard risk pillars")
    top_high_risk_sentences: List[SentenceRiskAnnotation] = Field(
        default_factory=list,
        description="Highest risk sentences flagged for analyst audit"
    )


# ------------------------------------------------------------------------------
# Generative AI Schemas
# ------------------------------------------------------------------------------
class GenAISummaryRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Financial text to synthesize")
    doc_type: Optional[DocumentType] = Field(default=DocumentType.GENERAL)
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific risk areas to focus on (e.g. ['Credit Risk', 'Liquidity'])"
    )


class ExecutiveRiskSummary(BaseModel):
    executive_summary: str = Field(..., description="High-level executive briefing on document risk posture")
    primary_concern: str = Field(..., description="Single most critical risk factor identified")
    key_risk_drivers: List[str] = Field(..., description="Bulleted list of primary risk drivers")
    hedges_or_stabilizers: List[str] = Field(default_factory=list, description="Positive compensating factors or mitigants noted")


class MitigationAction(BaseModel):
    timeframe: str = Field(..., description="Time horizon (e.g. Immediate 30-Day, 90-Day, Strategic)")
    action_title: str
    description: str
    target_risk_category: str
    priority: RiskLevel


class MitigationResponse(BaseModel):
    company_context: Optional[str]
    mitigation_actions: List[MitigationAction]
    governance_recommendations: List[str]


class RiskQARequest(BaseModel):
    document_text: str = Field(..., min_length=20, description="Financial document context")
    question: str = Field(
        ...,
        min_length=5,
        description="Question regarding risks, liabilities, covenants, or exposures",
        examples=["What are the primary regulatory challenges mentioned?"]
    )


class RiskQAResponse(BaseModel):
    question: str
    answer: str
    relevant_excerpts: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ------------------------------------------------------------------------------
# Full Unified Pipeline Response
# ------------------------------------------------------------------------------
class ComprehensiveAnalysisResponse(BaseModel):
    document_metadata: DocumentMetadata
    finbert_sentiment: FinBERTResponse
    bert_risk_profile: BERTRiskResponse
    executive_summary: Optional[ExecutiveRiskSummary] = None
    mitigation_plan: Optional[MitigationResponse] = None
    analysis_timestamp: str
    processing_time_ms: float


# ------------------------------------------------------------------------------
# Health & Status Schemas
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    device: str
    finbert_status: str
    bert_risk_status: str
    genai_engine: str
