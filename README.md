# 📊 Financial Document Risk Analysis API

**🔗 Live Demo:** [tdds022b-financialriskintelligence.streamlit.app](https://tdds022b-financialriskintelligence.streamlit.app/)

An enterprise-grade REST API for automated **Financial Document Risk Analysis** powered by **FastAPI**, **FinBERT**, **BERT**, and **Generative AI**. 

The backend automatically serves an interactive **Swagger UI** (`/docs`) and **ReDoc** (`/redoc`) documentation interface with built-in schema validation, interactive test runners, and sample payloads.

---

## 🎯 Key Features

1. **FinBERT Financial Tone & Polarity Engine**:
   - Analyzes disclosures using domain-adapted transformer architectures fine-tuned on financial corpora (*Financial PhraseBank* / *Loughran-McDonald*).
   - Computes discrete probabilities (`positive`, `neutral`, `negative`), a continuous **Sentiment Polarity Index** \([-1.0, +1.0]\), and a **Financial Uncertainty Score**.

2. **BERT 5-Pillar Financial Risk Taxonomy**:
   - Deep contextual classification across the 5 core financial risk pillars:
     - 💳 **Credit Risk**: Debt covenants, default vulnerabilities, counterparty solvency.
     - 📈 **Market Risk**: Interest rate fluctuations, FX currency swings, asset write-downs.
     - 💧 **Liquidity Risk**: Cash runway, working capital constraints, refinancing cliffs.
     - ⚙️ **Operational Risk**: Cyber attacks, system outages, supply chain bottlenecks.
     - ⚖️ **Regulatory & Legal Risk**: Antitrust litigation, SEC investigations, compliance penalties.
   - Evaluates risk severity ratings (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
   - Sentence-level clause auditing: flags and ranks the highest-risk clauses in filings.

3. **Generative AI Executive Intelligence**:
   - Synthesizes quantitative model outputs into an **Executive Risk Briefing**.
   - Generates structured, time-phased **Mitigation Roadmaps** (`30-Day Immediate`, `90-Day Medium-Term`, `Strategic Horizon`).
   - Grounded **Financial Risk Q&A Assistant** with document attribution.
   - Supports live **Google Gemini API** (`gemini-2.0-flash`) or zero-dependency built-in financial reasoning synthesis.

4. **Interactive Swagger UI Backend**:
   - Complete OpenAPI 3.1 specification.
   - Tagged endpoint categorization, syntax highlighting, schema introspection, and in-browser execution.

---

## 🏗️ Architecture

```
financial-risk-analysis-api/
├── app/
│   ├── main.py                       # FastAPI application & Swagger UI configuration
│   ├── config.py                     # Pydantic settings & environment loader
│   ├── api/
│   │   ├── deps.py                   # Dependency injection providers
│   │   └── v1/
│   │       ├── router.py             # Main API v1 router with Swagger tags
│   │       └── endpoints/
│   │           ├── health.py         # System health & engine readiness
│   │           ├── analyze.py        # Comprehensive 360° risk analysis (Text & Files)
│   │           ├── finbert.py        # FinBERT sentiment & uncertainty
│   │           ├── bert_risk.py      # BERT 5-pillar risk classification
│   │           └── generative.py     # Executive summary, mitigation, & Q&A
│   ├── models/
│   │   └── schemas.py                # Pydantic v2 schemas & OpenAPI examples
│   └── services/
│       ├── document_parser.py        # Text cleaning, chunking & file extractors
│       ├── finbert_service.py        # FinBERT inference engine (lazy loading + fallback)
│       ├── bert_service.py           # BERT risk classifier & clause ranking
│       ├── genai_service.py          # Generative AI (Gemini + fallback synthesis)
│       └── risk_pipeline.py          # Unified orchestration pipeline
├── sample_documents/
│   ├── sample_earnings_call.txt      # Real-world styled transcript
│   └── sample_10k_risk_section.txt   # Real-world styled 10-K Item 1A excerpt
├── streamlit_app.py                  # Interactive Streamlit Web UI Dashboard
├── run_ui.py                         # Streamlit UI launcher script
├── run_tests.py                      # Automated test suite (11 test cases)
├── run.py                            # FastAPI server launcher
├── requirements.txt                  # Python dependencies
└── .env.example                      # Configuration template
```

---

## 🚀 Quick Start

### 1. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional) To enable live Google Gemini LLM generation, set your API key in `.env`:*
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start the Backend API Server
```bash
python run.py
```
Once running locally, these become available in your own browser (they are **not** live links — they only work while the server is running on your machine):
- 🌐 **Swagger UI**: `http://localhost:8000/docs`
- 📖 **ReDoc**: `http://localhost:8000/redoc`

### 3. Launch the Streamlit Frontend Dashboard
In a new terminal:
```bash
python run_ui.py
```
Or with Streamlit directly:
```bash
streamlit run streamlit_app.py
```
Once running locally: 📊 **Streamlit Web UI**: `http://localhost:8501`

> **Note:** Want to try it without setting anything up? Use the hosted version instead: **https://tdds022b-financialriskintelligence.streamlit.app/**

---

## 📡 API Endpoints Reference

### 1. Full Document Risk Audit
- **`POST /api/v1/analyze/text`**: Full 360° risk audit on raw text disclosure.
- **`POST /api/v1/analyze/file`**: Multipart file upload (`.txt`, `.pdf`, `.json`, `.csv`) for end-to-end risk profiling.

#### Example Request (`POST /api/v1/analyze/text`):
```json
{
  "text": "Due to continuing supply chain disruptions and elevated foreign exchange volatility, operating margins declined by 320 basis points. Furthermore, ongoing antitrust litigation in Europe may result in material penalties.",
  "doc_type": "10-K",
  "company_name": "Apex Global Corp",
  "include_genai_summary": true,
  "include_mitigation": true
}
```

#### Example Response:
```json
{
  "document_metadata": {
    "total_characters": 221,
    "total_words": 30,
    "total_sentences": 2,
    "estimated_reading_time_mins": 0.15
  },
  "finbert_sentiment": {
    "dominant_sentiment": "negative",
    "sentiment_polarity_index": -0.739,
    "probabilities": {
      "positive": 0.0,
      "neutral": 0.261,
      "negative": 0.739
    },
    "financial_uncertainty_score": 0.3333,
    "interpretation": "Bearish / Risk-Elevated: High concentration of adverse financial disclosures (73.9% negative weight). Net polarity index: -0.74."
  },
  "bert_risk_profile": {
    "overall_risk_score": 0.56,
    "overall_risk_level": "HIGH",
    "categories": [
      {
        "category": "Market Risk",
        "score": 0.56,
        "risk_level": "HIGH",
        "key_factors": ["foreign exchange", "volatility"],
        "sample_clauses": ["Due to continuing supply chain disruptions and elevated foreign exchange volatility..."]
      },
      {
        "category": "Regulatory & Legal Risk",
        "score": 0.56,
        "risk_level": "HIGH",
        "key_factors": ["antitrust", "litigation"],
        "sample_clauses": ["Furthermore, ongoing antitrust litigation in Europe may result in material penalties."]
      }
    ],
    "top_high_risk_sentences": [
      {
        "sentence": "Furthermore, ongoing antitrust litigation in Europe may result in material penalties.",
        "risk_score": 0.56,
        "risk_level": "HIGH",
        "primary_category": "Regulatory & Legal Risk",
        "clause_index": 2
      }
    ]
  },
  "executive_summary": {
    "executive_summary": "The analyzed 10-K document exhibits significant risk aversion and negative sentiment (Sentiment Polarity Index: -0.74). The quantitative assessment flags Regulatory & Legal Risk as the dominant pressure point, warranting close monitoring by the enterprise risk committee.",
    "primary_concern": "Heightened exposure in Regulatory & Legal Risk: 'Furthermore, ongoing antitrust litigation in Europe may result in material penalties.'",
    "key_risk_drivers": [
      "Critical exposure identified across Market Risk (HIGH), Regulatory & Legal Risk (HIGH).",
      "Regulatory & Legal Risk drivers: antitrust, litigation."
    ],
    "hedges_or_stabilizers": [
      "Diversified revenue base or mitigating disclosure caveats noted in filing",
      "Ongoing liquidity monitoring and debt maturity management policies"
    ]
  },
  "mitigation_plan": {
    "company_context": "Apex Global Corp",
    "mitigation_actions": [
      {
        "timeframe": "Medium-Term (90 Days)",
        "action_title": "Regulatory Compliance & Litigation Pre-emption",
        "description": "Engage specialized regulatory counsel to audit antitrust and compliance disclosures and establish legal escrow reserves.",
        "target_risk_category": "Regulatory & Legal Risk",
        "priority": "HIGH"
      }
    ],
    "governance_recommendations": [
      "Mandate quarterly Board Audit Committee review of flagged risk metrics for Apex Global Corp.",
      "Incorporate dynamic debt covenant thresholds into monthly management reporting."
    ]
  },
  "analysis_timestamp": "2026-09-11T05:48:07.212000+00:00",
  "processing_time_ms": 11.5
}
```

---

### 2. Dedicated NLP Endpoints
- **`POST /api/v1/finbert/sentiment`**: Dedicated FinBERT financial sentiment, net polarity index, and disclosure uncertainty score.
- **`POST /api/v1/bert/risk-categories`**: Multi-category financial risk classification (Credit, Market, Liquidity, Operational, Regulatory).
- **`POST /api/v1/generative/summary`**: Generative AI executive risk summary.
- **`POST /api/v1/generative/mitigation`**: Structured 30-day, 90-day, and strategic mitigation plans.
- **`POST /api/v1/generative/qa`**: Context-grounded financial risk question-answering assistant.
- **`GET /api/v1/health`**: Hardware diagnostics, GPU/CUDA acceleration, and model readiness status.

---

## 🧪 Testing

Run the automated test suite covering all endpoints, parsers, and services:
```bash
python run_tests.py
```
Result:
```text
test_01_root_redirects_to_swagger ... ok
test_02_swagger_docs_html ... ok
test_03_openapi_schema ... ok
test_04_health_endpoint ... ok
test_05_finbert_sentiment_endpoint ... ok
test_06_bert_risk_categories_endpoint ... ok
test_07_generative_summary_endpoint ... ok
test_08_generative_mitigation_endpoint ... ok
test_09_generative_qa_endpoint ... ok
test_10_comprehensive_text_analysis ... ok
test_11_comprehensive_file_upload ... ok

----------------------------------------------------------------------
Ran 11 tests in 0.413s

OK
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API listening port |
| `DEBUG` | `True` | Hot-reload in development |
| `DEVICE` | `auto` | Execution device (`auto`, `cpu`, `cuda`) |
| `FINBERT_MODEL_NAME` | `ProsusAI/finbert` | FinBERT HuggingFace model repo |
| `BERT_RISK_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embedding model for risk pillars |
| `ENABLE_TRANSFORMER_DOWNLOAD` | `False` | Set `True` to allow HuggingFace Hub auto-downloads |
| `GEMINI_API_KEY` | `""` | Google Gemini API key for live LLM generation |
| `GEMINI_MODEL_NAME` | `gemini-2.0-flash` | Gemini model variant |
