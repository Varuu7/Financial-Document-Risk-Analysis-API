import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from app.config import get_settings
from app.models.schemas import (
    BERTRiskResponse,
    CategoryRiskScore,
    SentenceRiskAnnotation,
    RiskLevel
)
from app.services.document_parser import DocumentParser

logger = logging.getLogger("financial_api.bert_risk")


class BERTRiskClassifierService:
    """
    BERT-based Multi-Category Financial Risk Classification and Sentence Scoring.
    Evaluates documents across 5 pillars: Credit, Market, Liquidity, Operational, Regulatory & Legal Risk.
    """

    RISK_TAXONOMY = {
        "Credit Risk": {
            "anchors": [
                "debt default covenant breach bankruptcy credit rating downgrade",
                "non-accrual loans delinquent debt obligations credit loss reserves",
                "insolvency risk debtor non-performance counterparty default"
            ],
            "keywords": [
                "default", "covenant", "downgrade", "debt", "delinquent", "counterparty",
                "insolvent", "bankruptcy", "creditworthiness", "non-accrual", "credit rating"
            ]
        },
        "Market Risk": {
            "anchors": [
                "foreign exchange currency volatility interest rate fluctuation",
                "equity market declines commodity price inflation asset devaluation",
                "macroeconomic downturn valuation write-downs portfolio loss"
            ],
            "keywords": [
                "fx", "foreign exchange", "interest rate", "volatility", "inflation",
                "commodity", "devaluation", "mark-to-market", "equity markets", "macroeconomic"
            ]
        },
        "Liquidity Risk": {
            "anchors": [
                "cash runway deficit working capital shortage liquidity squeeze",
                "inability to refinance debt obligations revolving credit facility exhaustion",
                "cash flow constraint capital market access restrictions funding cliff"
            ],
            "keywords": [
                "liquidity", "cash flow", "working capital", "cash runway", "refinance",
                "credit facility", "burn rate", "shortfall", "funding", "insolvency"
            ]
        },
        "Operational Risk": {
            "anchors": [
                "cybersecurity breach ransomware attack IT infrastructure system outage",
                "supply chain disruption single source vendor dependency delivery delays",
                "loss of key personnel executive turnover internal controls deficiency"
            ],
            "keywords": [
                "cybersecurity", "ransomware", "data breach", "supply chain", "outage",
                "vendor", "turnover", "attrition", "disruption", "internal controls"
            ]
        },
        "Regulatory & Legal Risk": {
            "anchors": [
                "regulatory investigation compliance enforcement fines sanctions penalty",
                "pending material litigation lawsuit antitrust subpoena injunction",
                "statutory compliance violation patent infringement licensing revocation"
            ],
            "keywords": [
                "litigation", "lawsuit", "antitrust", "investigation", "subpoena",
                "fines", "penalty", "regulatory", "compliance", "sec", "doj", "injunction"
            ]
        }
    }

    def __init__(self):
        self.settings = get_settings()
        self._embedder = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self._category_anchor_embeddings: Dict[str, np.ndarray] = {}

    def get_status(self) -> str:
        if self._is_loaded:
            return f"Active (Embedding: {self.settings.bert_risk_model_name})"
        if self._load_error:
            return f"Fallback Mode (Heuristic Classifier: {self._load_error})"
        return "Initialized (Lazy Loading Enabled)"

    def _load_model(self):
        if self._is_loaded:
            return

        if not self.settings.enable_transformer_download:
            self._load_error = "Remote download disabled (ENABLE_TRANSFORMER_DOWNLOAD=False). Running zero-latency semantic risk classifier."
            logger.info(self._load_error)
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Downloading BERT Risk embedder: {self.settings.bert_risk_model_name}...")
            self._embedder = SentenceTransformer(self.settings.bert_risk_model_name)
            self._precompute_anchors()
            self._is_loaded = True
            logger.info("BERT Risk embedder loaded and initialized.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load SentenceTransformer ({e}). Using semantic keyword heuristics.")

    def _precompute_anchors(self):
        """Precomputes normalized category anchor vectors for fast cosine scoring."""
        for category, data in self.RISK_TAXONOMY.items():
            anchor_embs = self._embedder.encode(data["anchors"], convert_to_numpy=True)
            mean_emb = np.mean(anchor_embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            self._category_anchor_embeddings[category] = mean_emb / (norm if norm > 0 else 1.0)

    def _score_to_level(self, score: float) -> RiskLevel:
        if score >= 0.70:
            return RiskLevel.CRITICAL
        elif score >= 0.45:
            return RiskLevel.HIGH
        elif score >= 0.25:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def classify_risk(self, text: str) -> BERTRiskResponse:
        """Classifies document risks across the 5 dimensions and scores each sentence."""
        cleaned = DocumentParser.clean_text(text)
        if not cleaned:
            return BERTRiskResponse(
                overall_risk_score=0.0,
                overall_risk_level=RiskLevel.LOW,
                categories=[
                    CategoryRiskScore(category=cat, score=0.0, risk_level=RiskLevel.LOW, key_factors=[], sample_clauses=[])
                    for cat in self.RISK_TAXONOMY.keys()
                ],
                top_high_risk_sentences=[]
            )

        if not self._is_loaded and self._load_error is None:
            self._load_model()

        sentences = DocumentParser.segment_sentences(cleaned)
        if not sentences:
            sentences = [cleaned]

        sentence_annotations: List[SentenceRiskAnnotation] = []
        category_scores: Dict[str, float] = {cat: 0.0 for cat in self.RISK_TAXONOMY}
        category_factors: Dict[str, set] = {cat: set() for cat in self.RISK_TAXONOMY}
        category_clauses: Dict[str, list] = {cat: [] for cat in self.RISK_TAXONOMY}

        # Check if transformer embeddings are available
        if self._is_loaded and self._embedder is not None:
            sentence_annotations = self._classify_with_embeddings(
                sentences, category_scores, category_factors, category_clauses
            )
        else:
            sentence_annotations = self._classify_with_heuristics(
                sentences, category_scores, category_factors, category_clauses
            )

        # Build CategoryRiskScore list
        categories_output: List[CategoryRiskScore] = []
        for cat, score in category_scores.items():
            level = self._score_to_level(score)
            categories_output.append(
                CategoryRiskScore(
                    category=cat,
                    score=round(min(1.0, max(0.0, score)), 4),
                    risk_level=level,
                    key_factors=sorted(list(category_factors[cat]))[:6],
                    sample_clauses=category_clauses[cat][:2]
                )
            )

        # Global risk aggregation: weighted average giving higher weight to maximum risk pillar
        cat_score_values = [c.score for c in categories_output]
        max_score = max(cat_score_values) if cat_score_values else 0.0
        avg_score = (sum(cat_score_values) / len(cat_score_values)) if cat_score_values else 0.0
        overall_score = round(0.65 * max_score + 0.35 * avg_score, 4)
        overall_level = self._score_to_level(overall_score)

        # Top flagged high risk sentences
        high_risk_sentences = [
            s for s in sentence_annotations if s.risk_score >= 0.35
        ]
        high_risk_sentences.sort(key=lambda x: x.risk_score, reverse=True)

        return BERTRiskResponse(
            overall_risk_score=overall_score,
            overall_risk_level=overall_level,
            categories=categories_output,
            top_high_risk_sentences=high_risk_sentences[:10]
        )

    def _classify_with_embeddings(
        self,
        sentences: List[str],
        category_scores: Dict[str, float],
        category_factors: Dict[str, set],
        category_clauses: Dict[str, list]
    ) -> List[SentenceRiskAnnotation]:
        annotations: List[SentenceRiskAnnotation] = []
        
        # Limit batch size for high efficiency
        sample_sentences = sentences[:100]
        try:
            sent_embs = self._embedder.encode(sample_sentences, convert_to_numpy=True)
            # Normalize sentence embeddings
            norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sent_embs = sent_embs / norms

            for idx, (sentence, emb) in enumerate(zip(sample_sentences, sent_embs)):
                best_cat = "Operational Risk"
                best_sim = -1.0
                sentence_lower = sentence.lower()

                # Cosine similarity against each category anchor
                for cat, anchor_vec in self._category_anchor_embeddings.items():
                    sim = float(np.dot(emb, anchor_vec))
                    
                    # Boost with domain keywords
                    kw_hits = [kw for kw in self.RISK_TAXONOMY[cat]["keywords"] if kw in sentence_lower]
                    if kw_hits:
                        sim += len(kw_hits) * 0.08
                        category_factors[cat].update(kw_hits)

                    # Update category max tracking
                    if sim > category_scores[cat]:
                        category_scores[cat] = sim

                    if sim > best_sim:
                        best_sim = sim
                        best_cat = cat

                calibrated_score = round(min(1.0, max(0.0, (best_sim - 0.25) / 0.65)), 4)
                level = self._score_to_level(calibrated_score)

                if calibrated_score > 0.40 and len(category_clauses[best_cat]) < 3:
                    category_clauses[best_cat].append(sentence)

                annotations.append(
                    SentenceRiskAnnotation(
                        sentence=sentence,
                        risk_score=calibrated_score,
                        risk_level=level,
                        primary_category=best_cat,
                        clause_index=idx + 1
                    )
                )

            # Normalize category scores
            for cat in category_scores:
                category_scores[cat] = round(min(1.0, max(0.0, (category_scores[cat] - 0.25) / 0.65)), 4)

        except Exception as e:
            logger.warning(f"Error in embedding classification: {e}. Fallback to heuristics.")
            return self._classify_with_heuristics(sentences, category_scores, category_factors, category_clauses)

        return annotations

    def _classify_with_heuristics(
        self,
        sentences: List[str],
        category_scores: Dict[str, float],
        category_factors: Dict[str, set],
        category_clauses: Dict[str, list]
    ) -> List[SentenceRiskAnnotation]:
        annotations: List[SentenceRiskAnnotation] = []

        for idx, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            best_cat = "Operational Risk"
            best_score = 0.0

            for cat, data in self.RISK_TAXONOMY.items():
                kw_hits = [kw for kw in data["keywords"] if kw in sentence_lower]
                if kw_hits:
                    cat_score = min(0.95, len(kw_hits) * 0.28)
                    category_factors[cat].update(kw_hits)
                    category_scores[cat] = max(category_scores[cat], cat_score)
                    if cat_score > best_score:
                        best_score = cat_score
                        best_cat = cat

            level = self._score_to_level(best_score)
            if best_score > 0.35 and len(category_clauses[best_cat]) < 3:
                category_clauses[best_cat].append(sentence)

            annotations.append(
                SentenceRiskAnnotation(
                    sentence=sentence,
                    risk_score=round(best_score, 4),
                    risk_level=level,
                    primary_category=best_cat,
                    clause_index=idx + 1
                )
            )

        return annotations
