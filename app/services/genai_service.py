import json
import logging
import re
from typing import Dict, List, Optional, Any
import httpx
from app.config import get_settings
from app.models.schemas import (
    ExecutiveRiskSummary,
    MitigationResponse,
    MitigationAction,
    RiskQAResponse,
    RiskLevel,
    FinBERTResponse,
    BERTRiskResponse
)
from app.services.document_parser import DocumentParser

logger = logging.getLogger("financial_api.genai")


class GenerativeAIService:
    """
    Generative AI Service for Financial Risk Analysis.
    Supports Google Gemini API with seamless fallback to an offline financial synthesis engine.
    """

    def __init__(self):
        self.settings = get_settings()

    def get_status(self) -> str:
        if self.settings.gemini_api_key:
            masked = self.settings.gemini_api_key[:4] + "..." + self.settings.gemini_api_key[-4:]
            return f"Active (Gemini Model: {self.settings.gemini_model_name}, Key: {masked})"
        return "Active (Built-in Financial Risk Synthesis Engine - Set GEMINI_API_KEY in .env for live LLM)"

    async def generate_executive_summary(
        self,
        text: str,
        doc_type: str = "general",
        finbert_result: Optional[FinBERTResponse] = None,
        bert_result: Optional[BERTRiskResponse] = None
    ) -> ExecutiveRiskSummary:
        """Generates a comprehensive executive risk briefing."""
        # Try live Gemini API if key is present
        if self.settings.gemini_api_key:
            try:
                summary = await self._call_gemini_summary(text, doc_type, finbert_result, bert_result)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"Gemini API error ({e}). Falling back to built-in synthesis.")

        # Built-in structured synthesis engine
        return self._synthesize_executive_summary(text, doc_type, finbert_result, bert_result)

    async def generate_mitigation_plan(
        self,
        text: str,
        company_name: Optional[str] = None,
        bert_result: Optional[BERTRiskResponse] = None
    ) -> MitigationResponse:
        """Generates prioritized risk mitigation and governance recommendations."""
        if self.settings.gemini_api_key:
            try:
                plan = await self._call_gemini_mitigation(text, company_name, bert_result)
                if plan:
                    return plan
            except Exception as e:
                logger.warning(f"Gemini API error ({e}). Falling back to built-in mitigation.")

        return self._synthesize_mitigation_plan(text, company_name, bert_result)

    async def answer_question(
        self,
        document_text: str,
        question: str
    ) -> RiskQAResponse:
        """Answers financial queries grounded in document evidence."""
        if self.settings.gemini_api_key:
            try:
                res = await self._call_gemini_qa(document_text, question)
                if res:
                    return res
            except Exception as e:
                logger.warning(f"Gemini QA error ({e}). Falling back to local search QA.")

        return self._synthesize_qa(document_text, question)

    # --------------------------------------------------------------------------
    # Gemini API Live Callers
    # --------------------------------------------------------------------------
    async def _call_gemini_summary(
        self,
        text: str,
        doc_type: str,
        finbert_result: Optional[FinBERTResponse],
        bert_result: Optional[BERTRiskResponse]
    ) -> Optional[ExecutiveRiskSummary]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model_name}:generateContent?key={self.settings.gemini_api_key}"
        )
        prompt = (
            f"You are a Senior Financial Risk Officer. Analyze the following {doc_type} disclosure.\n\n"
            f"Document Excerpt:\n{text[:4000]}\n\n"
            f"Quant Metrics: FinBERT Dominant Sentiment: {finbert_result.dominant_sentiment if finbert_result else 'N/A'}, "
            f"Overall BERT Risk Level: {bert_result.overall_risk_level.value if bert_result else 'N/A'}.\n\n"
            f"Return a strict valid JSON object with the following fields:\n"
            f'{{"executive_summary": "<paragraph>", "primary_concern": "<sentence>", "key_risk_drivers": ["<item1>", "<item2>"], "hedges_or_stabilizers": ["<item1>"]}}'
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(content)
                return ExecutiveRiskSummary(**parsed)
        return None

    async def _call_gemini_mitigation(
        self,
        text: str,
        company_name: Optional[str],
        bert_result: Optional[BERTRiskResponse]
    ) -> Optional[MitigationResponse]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model_name}:generateContent?key={self.settings.gemini_api_key}"
        )
        prompt = (
            f"You are a Financial Risk Consultant. Propose mitigation actions for {company_name or 'the entity'}.\n\n"
            f"Context:\n{text[:4000]}\n\n"
            f"Provide a strict JSON object with fields:\n"
            f'{{"company_context": "{company_name or "Entity"}", "mitigation_actions": [{{"timeframe": "Immediate (30 Days)", "action_title": "title", "description": "desc", "target_risk_category": "Credit Risk", "priority": "HIGH"}}], "governance_recommendations": ["rec1"]}}'
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(content)
                return MitigationResponse(**parsed)
        return None

    async def _call_gemini_qa(self, document_text: str, question: str) -> Optional[RiskQAResponse]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model_name}:generateContent?key={self.settings.gemini_api_key}"
        )
        prompt = (
            f"Context Document:\n{document_text[:5000]}\n\n"
            f"Question: {question}\n\n"
            f"Answer solely based on the context. Return JSON:\n"
            f'{{"question": "{question}", "answer": "<answer>", "relevant_excerpts": ["<quote1>"], "confidence": 0.95}}'
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(content)
                return RiskQAResponse(**parsed)
        return None

    # --------------------------------------------------------------------------
    # Built-in Financial Risk Reasoning Synthesis (Zero-dependency Fallback)
    # --------------------------------------------------------------------------
    def _synthesize_executive_summary(
        self,
        text: str,
        doc_type: str,
        finbert_result: Optional[FinBERTResponse],
        bert_result: Optional[BERTRiskResponse]
    ) -> ExecutiveRiskSummary:
        """Synthesizes an analyst-grade executive risk briefing."""
        top_category = "General Business Risk"
        highest_score = 0.0
        high_risk_categories = []

        if bert_result:
            for cat in bert_result.categories:
                if cat.score > highest_score:
                    highest_score = cat.score
                    top_category = cat.category
                if cat.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                    high_risk_categories.append(f"{cat.category} ({cat.risk_level.value})")

        tone = finbert_result.dominant_sentiment if finbert_result else "neutral"
        polarity = finbert_result.sentiment_polarity_index if finbert_result else 0.0

        # Construct primary concern
        if bert_result and bert_result.top_high_risk_sentences:
            flagged = bert_result.top_high_risk_sentences[0].sentence
            primary_concern = f"Heightened exposure in {top_category}: '{flagged[:140]}...'"
        else:
            primary_concern = f"Primary vulnerability centered around {top_category} with elevated uncertainty."

        # Construct drivers
        drivers = []
        if high_risk_categories:
            drivers.append(f"Critical exposure identified across {', '.join(high_risk_categories)}.")
        if finbert_result and finbert_result.financial_uncertainty_score > 0.35:
            drivers.append(f"Elevated disclosure uncertainty ({finbert_result.financial_uncertainty_score * 100:.0f}%) reflecting ambiguous operating conditions.")
        if bert_result:
            for cat in bert_result.categories:
                if cat.key_factors:
                    drivers.append(f"{cat.category} drivers: {', '.join(cat.key_factors[:3])}.")

        if not drivers:
            drivers.append("Standard macro and operational headwinds common to financial disclosures.")

        # Construct executive summary text
        sentiment_phrase = (
            "significant risk aversion and negative sentiment" if tone == "negative"
            else "positive operating momentum with manageable risk exposures" if tone == "positive"
            else "a balanced posture with specific pockets of operational and market friction"
        )

        exec_text = (
            f"The analyzed {doc_type} document exhibits {sentiment_phrase} "
            f"(Sentiment Polarity Index: {polarity:+.2f}). "
            f"The quantitative assessment flags {top_category} as the dominant pressure point, "
            f"warranting close monitoring by the enterprise risk committee."
        )

        # Hedges / Stabilizers
        hedges = [
            "Diversified revenue base or mitigating disclosure caveats noted in filing",
            "Ongoing liquidity monitoring and debt maturity management policies"
        ]

        return ExecutiveRiskSummary(
            executive_summary=exec_text,
            primary_concern=primary_concern,
            key_risk_drivers=drivers[:5],
            hedges_or_stabilizers=hedges
        )

    def _synthesize_mitigation_plan(
        self,
        text: str,
        company_name: Optional[str],
        bert_result: Optional[BERTRiskResponse]
    ) -> MitigationResponse:
        """Generates structured time-horizon mitigation actions."""
        target_company = company_name or "Reporting Entity"
        actions: List[MitigationAction] = []

        # Find high priority categories
        high_cats = []
        if bert_result:
            for cat in bert_result.categories:
                if cat.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                    high_cats.append(cat.category)

        # Immediate Actions (30 Days)
        if "Credit Risk" in high_cats or "Liquidity Risk" in high_cats:
            actions.append(
                MitigationAction(
                    timeframe="Immediate (30 Days)",
                    action_title="Liquidity Runway & Debt Covenant Stress-Testing",
                    description="Conduct immediate sensitivity analysis on EBITDA coverage covenants and establish contingency cash reserves.",
                    target_risk_category="Liquidity / Credit Risk",
                    priority=RiskLevel.CRITICAL
                )
            )
        else:
            actions.append(
                MitigationAction(
                    timeframe="Immediate (30 Days)",
                    action_title="Comprehensive Exposure Audit",
                    description="Establish cross-functional risk monitoring team to audit disclosed risk factors against live Q3/Q4 KPIs.",
                    target_risk_category="Operational Risk",
                    priority=RiskLevel.HIGH
                )
            )

        # Medium-Term Actions (90 Days)
        if "Regulatory & Legal Risk" in high_cats:
            actions.append(
                MitigationAction(
                    timeframe="Medium-Term (90 Days)",
                    action_title="Regulatory Compliance & Litigation Pre-emption",
                    description="Engage specialized regulatory counsel to audit antitrust and compliance disclosures and establish legal escrow reserves.",
                    target_risk_category="Regulatory & Legal Risk",
                    priority=RiskLevel.HIGH
                )
            )
        else:
            actions.append(
                MitigationAction(
                    timeframe="Medium-Term (90 Days)",
                    action_title="Market Volatility & FX Hedging Framework",
                    description="Implement derivative hedges (collars/swaps) to insulate operating margins against benchmark interest and currency swings.",
                    target_risk_category="Market Risk",
                    priority=RiskLevel.MEDIUM
                )
            )

        # Strategic Actions (Strategic Horizon)
        actions.append(
            MitigationAction(
                timeframe="Strategic (1 Year+)",
                action_title="Enterprise Risk Architecture & Supply Chain Redundancy",
                description="Diversify critical supplier bases and institutionalize AI-driven continuous risk surveillance into internal audit.",
                target_risk_category="Operational Risk",
                priority=RiskLevel.MEDIUM
            )
        )

        governance = [
            f"Mandate quarterly Board Audit Committee review of flagged risk metrics for {target_company}.",
            "Incorporate dynamic debt covenant thresholds into monthly management reporting.",
            "Enforce vendor redundancy requirements for single-source operational dependencies."
        ]

        return MitigationResponse(
            company_context=target_company,
            mitigation_actions=actions,
            governance_recommendations=governance
        )

    def _synthesize_qa(self, text: str, question: str) -> RiskQAResponse:
        """Grounded Q&A heuristic finder."""
        sentences = DocumentParser.segment_sentences(text)
        q_words = set(re.findall(r'\b\w{3,}\b', question.lower()))

        scored_sentences = []
        for s in sentences:
            s_lower = s.lower()
            overlap = sum(1 for w in q_words if w in s_lower)
            if overlap > 0:
                scored_sentences.append((overlap, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_excerpts = [s for _, s in scored_sentences[:3]]

        if top_excerpts:
            answer = (
                f"Based on the document context, the following relevant facts address your inquiry: "
                f"{' '.join(top_excerpts)}"
            )
            confidence = 0.85
        else:
            answer = "No direct clause in the provided document specifically matched the key terms of your question."
            confidence = 0.30

        return RiskQAResponse(
            question=question,
            answer=answer,
            relevant_excerpts=top_excerpts,
            confidence=confidence
        )
