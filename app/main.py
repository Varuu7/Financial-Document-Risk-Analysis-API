import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from app.config import get_settings
from app.api.v1.router import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("financial_api")

settings = get_settings()

# Swagger UI Tags Metadata
tags_metadata = [
    {
        "name": "Health & System Status",
        "description": "System diagnostics, hardware acceleration (CPU/CUDA), and engine statuses.",
    },
    {
        "name": "Full Document Risk Audit",
        "description": "Unified 360-degree risk auditing combining FinBERT, BERT, and Generative AI for text and files.",
    },
    {
        "name": "FinBERT Financial Sentiment",
        "description": "Specialized FinBERT transformer inference for financial tone, polarity index, and disclosure uncertainty.",
    },
    {
        "name": "BERT Multi-Category Risk Engine",
        "description": "Deep contextual risk categorization across Credit, Market, Liquidity, Operational, and Legal risk pillars.",
    },
    {
        "name": "Generative AI Intelligence",
        "description": "Executive risk synthesis, time-phased mitigation roadmaps, and context-grounded risk Q&A.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown routines."""
    logger.info("==========================================================")
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Swagger UI available at: http://{settings.host}:{settings.port}/docs")
    logger.info("==========================================================")
    yield
    logger.info("Shutting down Financial Document Risk Analysis API.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 2,
        "docExpansion": "list",
        "filter": True,
        "persistAuthorization": True,
        "syntaxHighlight.theme": "monokai"
    },
    lifespan=lifespan
)

# Enable CORS for cross-origin client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse, FileResponse
from pathlib import Path

# Mount API v1 router
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path directly to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/download",
    summary="Download Project ZIP",
    description="Direct browser download link for the entire project source code archive.",
    tags=["Health & System Status"]
)
async def download_project_zip():
    """Provides a direct browser download of the complete project zip archive."""
    zip_path = Path(r"C:\Users\Varun\.gemini\antigravity\scratch\financial-risk-analysis-api.zip")
    if not zip_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Zip file not found")
    return FileResponse(
        path=str(zip_path),
        filename="financial-risk-analysis-api.zip",
        media_type="application/zip"
    )
