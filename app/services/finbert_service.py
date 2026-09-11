import re
import logging
from typing import Dict, List, Optional
import torch
from app.config import get_settings
from app.models.schemas import FinBERTResponse, FinBERTProbabilities
from app.services.document_parser import DocumentParser

logger = logging.getLogger("financial_api.finbert")


class FinBERTService:
    """
    FinBERT Financial Sentiment & Tone Inference Engine.
    Uses Hugging Face's ProsusAI/finbert transformer model fine-tuned on Financial PhraseBank.
    Includes automated fallback to Loughran-McDonald financial lexicon when weights are loading/offline.
    """

    # Loughran-McDonald inspired financial risk & sentiment lexicons
    FINANCIAL_NEGATIVE_WORDS = {
        "loss", "losses", "decline", "declined", "declining", "drop", "dropped", "deficit",
        "adverse", "adversely", "impairment", "impaired", "impairments", "default", "defaulted", "breach",
        "breached", "penalties", "penalty", "litigation", "lawsuit", "investigation", "subpoena",
        "insolvency", "bankruptcy", "deterioration", "downturn", "restructuring", "severance",
        "curtailment", "headwind", "headwinds", "volatility", "risk", "risks", "write-down",
        "write-off", "curtail", "terminate", "terminated", "violation", "distress", "weakness",
        "contracted", "contraction", "catastrophic", "severe", "severely"
    }

    FINANCIAL_POSITIVE_WORDS = {
        "growth", "grow", "grew", "increase", "increased", "gain", "gains", "profit",
        "profitable", "profitability", "exceeded", "outperform", "outperformed", "dividend",
        "expansion", "expanding", "rebound", "strengthened", "strong", "resilient", "resilience",
        "tailwind", "tailwinds", "upside", "record", "accretive", "robust", "momentum", "surplus"
    }

    FINANCIAL_UNCERTAINTY_WORDS = {
        "may", "might", "could", "possibly", "potential", "potentially", "uncertain", "uncertainty",
        "uncertainties", "approximate", "approximately", "contingent", "fluctuate", "fluctuations",
        "subject to", "anticipate", "anticipated", "believe", "estimate", "estimated", "forecast"
    }

    def __init__(self):
        self.settings = get_settings()
        self._pipeline = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
        self.device = self._determine_device()

    def _determine_device(self) -> str:
        if self.settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.settings.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def is_model_loaded(self) -> bool:
        return self._is_loaded

    def get_status(self) -> str:
        if self._is_loaded:
            return f"Active (Transformer: {self.settings.finbert_model_name} on {self.device})"
        if self._load_error:
            return f"Fallback Mode (Lexicon-Engine: {self._load_error})"
        return "Initialized (Lazy Loading Enabled)"

    def _load_model(self):
        """Lazy loader for FinBERT transformer pipeline."""
        if self._is_loaded:
            return

        if not self.settings.enable_transformer_download:
            self._load_error = "Remote download disabled (ENABLE_TRANSFORMER_DOWNLOAD=False). Running zero-latency financial lexicon engine."
            logger.info(self._load_error)
            return

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            device_id = 0 if self.device == "cuda" else -1

            logger.info(f"Downloading FinBERT model: {self.settings.finbert_model_name} on {self.device}...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.settings.finbert_model_name,
                device=device_id,
                top_k=None,
                truncation=True,
                max_length=512
            )
            self._is_loaded = True
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Could not load FinBERT transformer ({e}). Operating in resilient financial lexicon mode.")

    def analyze(self, text: str) -> FinBERTResponse:
        """Analyzes financial sentiment, polarity, and uncertainty of the input text."""
        cleaned_text = DocumentParser.clean_text(text)
        if not cleaned_text:
            return FinBERTResponse(
                dominant_sentiment="neutral",
                sentiment_polarity_index=0.0,
                probabilities=FinBERTProbabilities(positive=0.33, neutral=0.34, negative=0.33),
                financial_uncertainty_score=0.0,
                interpretation="Empty document provided; default neutral stance assigned."
            )

        # Attempt to load transformer if not yet attempted
        if not self._is_loaded and self._load_error is None:
            self._load_model()

        # Compute uncertainty score regardless of model backend
        uncertainty_score = self._compute_uncertainty_score(cleaned_text)

        # Transformer Inference
        if self._is_loaded and self._pipeline is not None:
            try:
                sentences = DocumentParser.segment_sentences(cleaned_text)
                if not sentences:
                    sentences = [cleaned_text[:512]]
                
                # Sample up to top 20 representative sentences for batch inference efficiency
                sample_sentences = sentences[:20]
                results = self._pipeline(sample_sentences)

                # Aggregate probabilities across sentences
                agg_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                for item in results:
                    for score_dict in item:
                        label = score_dict["label"].lower()
                        if label in agg_scores:
                            agg_scores[label] += score_dict["score"]

                total = sum(agg_scores.values()) or 1.0
                pos_prob = round(agg_scores["positive"] / total, 4)
                neu_prob = round(agg_scores["neutral"] / total, 4)
                neg_prob = round(agg_scores["negative"] / total, 4)

                return self._format_response(pos_prob, neu_prob, neg_prob, uncertainty_score)
            except Exception as e:
                logger.warning(f"Transformer inference error: {e}. Utilizing fallback scoring.")

        # Fallback Lexicon-based Financial Sentiment Engine
        return self._lexicon_analysis(cleaned_text, uncertainty_score)

    def _compute_uncertainty_score(self, text: str) -> float:
        """Calculates ratio of financial hedging and uncertainty expressions."""
        words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        if not words:
            return 0.0
        uncertainty_matches = sum(1 for w in words if w in self.FINANCIAL_UNCERTAINTY_WORDS)
        # Ratio scaled with sigmoid-like curve
        ratio = (uncertainty_matches / len(words)) * 10.0
        return round(min(1.0, max(0.0, ratio)), 4)

    def _lexicon_analysis(self, text: str, uncertainty_score: float) -> FinBERTResponse:
        """High-precision financial sentiment scorer using calibrated domain vocabularies."""
        words = re.findall(r'\b[a-zA-Z-]+\b', text.lower())
        if not words:
            return self._format_response(0.33, 0.34, 0.33, uncertainty_score)

        neg_count = sum(1 for w in words if w in self.FINANCIAL_NEGATIVE_WORDS)
        pos_count = sum(1 for w in words if w in self.FINANCIAL_POSITIVE_WORDS)

        total_sentiment_words = neg_count + pos_count
        if total_sentiment_words == 0:
            return self._format_response(0.10, 0.80, 0.10, uncertainty_score)

        neg_ratio = neg_count / total_sentiment_words
        pos_ratio = pos_count / total_sentiment_words

        # Neutral weight diminishes as sentiment signal concentration increases
        neutral_weight = max(0.15, 0.55 - 0.12 * total_sentiment_words)
        sentiment_mass = 1.0 - neutral_weight

        pos_prob = round(pos_ratio * sentiment_mass, 4)
        neg_prob = round(neg_ratio * sentiment_mass, 4)
        neu_prob = round(neutral_weight, 4)

        return self._format_response(pos_prob, neu_prob, neg_prob, uncertainty_score)

    def _format_response(self, pos: float, neu: float, neg: float, uncertainty: float) -> FinBERTResponse:
        polarity = round(pos - neg, 4)
        
        if neg > pos and neg > neu:
            dominant = "negative"
            interp = (
                f"Bearish / Risk-Elevated: High concentration of adverse financial disclosures "
                f"({neg * 100:.1f}% negative weight). Net polarity index: {polarity:+.2f}."
            )
        elif pos > neg and pos > neu:
            dominant = "positive"
            interp = (
                f"Bullish / Favorable: Predominantly accretive disclosures "
                f"({pos * 100:.1f}% positive weight). Net polarity index: {polarity:+.2f}."
            )
        else:
            dominant = "neutral"
            interp = (
                f"Neutral / Informational: Balanced or factual disclosures with {neu * 100:.1f}% neutrality. "
                f"Net polarity index: {polarity:+.2f}."
            )

        if uncertainty > 0.40:
            interp += f" Elevated financial uncertainty detected ({uncertainty * 100:.1f}%)."

        return FinBERTResponse(
            dominant_sentiment=dominant,
            sentiment_polarity_index=polarity,
            probabilities=FinBERTProbabilities(
                positive=pos,
                neutral=neu,
                negative=neg
            ),
            financial_uncertainty_score=uncertainty,
            interpretation=interp
        )
