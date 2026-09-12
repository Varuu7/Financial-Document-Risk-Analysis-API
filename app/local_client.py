"""
In-process client for the Streamlit UI.

streamlit_app.py originally talked to the FastAPI backend over HTTP
(http://127.0.0.1:8000/...), which requires a separate `uvicorn` server to be
running. On single-container deployments (like Streamlit Community Cloud)
that meant either running a second process or spinning up FastAPI in a
background thread - both of which add real failure modes (port conflicts,
startup races, resource limits) on top of the actual analysis work.

This module exposes `get(path, ...)` / `post(path, ...)` functions with the
same call signature and return shape (`.status_code`, `.json()`, `.text`)
as the `requests` library, but they call the underlying services directly in
the same process - no HTTP, no separate server, no port to be "unreachable."

`run.py` / `app.main` still exist unchanged for anyone who wants to run the
FastAPI server standalone (e.g. to hit it from Postman, curl, or another
client) - this module is purely an additional, simpler path for the
Streamlit UI.
"""
import asyncio
import json as _json

from app.config import get_settings
from app.api.deps import (
    get_finbert_service,
    get_bert_service,
    get_genai_service,
    get_risk_pipeline,
)
from app.models.schemas import TextAnalysisRequest, HealthResponse
from app.services.document_parser import DocumentParser


def _run_async(coro):
    return asyncio.run(coro)


def _pipeline():
    return get_risk_pipeline(
        get_finbert_service(), get_bert_service(), get_genai_service()
    )


class _FakeResponse:
    """Mimics the small subset of requests.Response that streamlit_app.py uses."""

    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (_json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload


def _error_response(exc: Exception) -> _FakeResponse:
    return _FakeResponse(500, text=str(exc))


def _path_of(url: str) -> str:
    # Callers pass full URLs like "http://127.0.0.1:8000/api/v1/health" so
    # existing call sites in streamlit_app.py don't need to change beyond
    # swapping `requests` for this module - we only care about the path.
    if "://" in url:
        url = url.split("://", 1)[1]
    slash = url.find("/")
    return url[slash:] if slash != -1 else "/"


def get(url: str, timeout=None):
    path = _path_of(url)
    try:
        if path == "/api/v1/health":
            settings = get_settings()
            finbert = get_finbert_service()
            bert = get_bert_service()
            genai = get_genai_service()

            device_name = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device_name = f"cuda ({torch.cuda.get_device_name(0)})"
            except Exception:
                pass

            resp = HealthResponse(
                status="healthy",
                app_name=settings.app_name,
                version=settings.app_version,
                device=device_name,
                finbert_status=finbert.get_status(),
                bert_risk_status=bert.get_status(),
                genai_engine=genai.get_status(),
            )
            return _FakeResponse(200, resp.model_dump(mode="json"))

        return _FakeResponse(404, text=f"Unknown endpoint: {path}")
    except Exception as e:
        return _error_response(e)


def post(url: str, json=None, data=None, files=None, timeout=None):
    path = _path_of(url)
    try:
        if path == "/api/v1/analyze/text":
            payload = json or {}
            request = TextAnalysisRequest(**payload)
            result = _run_async(_pipeline().run(request))
            return _FakeResponse(200, result.model_dump(mode="json"))

        if path == "/api/v1/analyze/file":
            filename, content, _mimetype = files["file"]
            if isinstance(content, str):
                content = content.encode("utf-8")
            extracted_text = DocumentParser.parse_uploaded_file(filename, content)

            if len(extracted_text.strip()) < 20:
                return _FakeResponse(
                    400,
                    text=(
                        "Extracted document content is too short for financial "
                        "risk analysis (minimum 20 characters required)."
                    ),
                )

            form = data or {}
            request = TextAnalysisRequest(
                text=extracted_text,
                doc_type=form.get("doc_type", "general"),
                company_name=form.get("company_name") or None,
                include_genai_summary=str(
                    form.get("include_genai_summary", "true")
                ).lower()
                == "true",
                include_mitigation=str(form.get("include_mitigation", "true")).lower()
                == "true",
            )
            result = _run_async(_pipeline().run(request))
            return _FakeResponse(200, result.model_dump(mode="json"))

        if path == "/api/v1/generative/qa":
            payload = json or {}
            genai = get_genai_service()
            result = _run_async(
                genai.answer_question(
                    document_text=payload.get("document_text", ""),
                    question=payload.get("question", ""),
                )
            )
            return _FakeResponse(200, result.model_dump(mode="json"))

        if path == "/api/v1/finbert/sentiment":
            payload = json or {}
            finbert = get_finbert_service()
            result = finbert.analyze(payload.get("text", ""))
            return _FakeResponse(200, result.model_dump(mode="json"))

        if path == "/api/v1/bert/risk-categories":
            payload = json or {}
            bert = get_bert_service()
            result = bert.classify_risk(payload.get("text", ""))
            return _FakeResponse(200, result.model_dump(mode="json"))

        return _FakeResponse(404, text=f"Unknown endpoint: {path}")
    except Exception as e:
        return _error_response(e)
