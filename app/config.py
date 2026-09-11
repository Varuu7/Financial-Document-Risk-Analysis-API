import os
from pathlib import Path
from functools import lru_cache
from typing import Literal
from pydantic import BaseModel, Field

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Lightweight .env loader if python-dotenv is not installed
def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key not in os.environ:
                os.environ[key] = val

load_env_file(BASE_DIR / ".env")


class Settings(BaseModel):
    """Application settings and configuration."""
    app_name: str = "Financial Document Risk Analysis API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Enterprise-grade Financial Document Risk Analysis API powered by FinBERT, "
        "BERT embeddings, and Generative AI. Features automated executive summaries, "
        "multi-category risk classification (Credit, Market, Liquidity, Operational, Regulatory), "
        "sentence-level risk heatmaps, and mitigation strategies."
    )
    api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/api/v1")
    debug: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Host & Port
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Device: auto, cpu, cuda
    device: str = os.getenv("DEVICE", "auto")

    # Transformer Models
    finbert_model_name: str = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
    bert_risk_model_name: str = os.getenv("BERT_RISK_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    enable_transformer_download: bool = os.getenv("ENABLE_TRANSFORMER_DOWNLOAD", "False").lower() in ("true", "1", "yes")

    # Generative AI (Gemini / Fallback)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

    # Pipeline Settings
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
