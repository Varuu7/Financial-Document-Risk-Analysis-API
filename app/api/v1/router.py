from fastapi import APIRouter
from app.api.v1.endpoints import health, analyze, finbert, bert_risk, generative

api_router = APIRouter()

# Register endpoint routers with descriptive Swagger UI tags
api_router.include_router(health.router, tags=["Health & System Status"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["Full Document Risk Audit"])
api_router.include_router(finbert.router, prefix="/finbert", tags=["FinBERT Financial Sentiment"])
api_router.include_router(bert_risk.router, prefix="/bert", tags=["BERT Multi-Category Risk Engine"])
api_router.include_router(generative.router, prefix="/generative", tags=["Generative AI Intelligence"])
