import os
from pathlib import Path
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Risk Intelligence | FinBERT & GenAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Custom CSS for Financial Terminal Aesthetics
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Severity Badges */
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* Executive Briefing Box */
    .exec-box {
        background-color: #1e1e2e;
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Constants & Helper Functions
# ------------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_documents"


@st.cache_data(ttl=5)
def check_backend_health(api_url: str):
    try:
        r = requests.get(f"{api_url}/api/v1/health", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


def load_sample_file(filename: str) -> str:
    path = SAMPLE_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Initialize session state for document text
if "document_text" not in st.session_state:
    st.session_state.document_text = load_sample_file("sample_10k_risk_section.txt")

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/combo-chart.png", width=64)
    st.title("Financial Risk AI")
    st.caption("FinBERT • BERT • Generative AI")

    st.markdown("---")
    st.subheader("🔌 Backend Status")
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL)
    is_healthy, health_info = check_backend_health(api_url)

    if is_healthy:
        st.success(f"● Connected ({health_info.get('device', 'cpu')})")
        with st.expander("Diagnostic Info"):
            st.write(f"**App:** {health_info.get('app_name')}")
            st.write(f"**FinBERT:** {health_info.get('finbert_status')}")
            st.write(f"**BERT Risk:** {health_info.get('bert_risk_status')}")
            st.write(f"**GenAI:** {health_info.get('genai_engine')}")
    else:
        st.error("● Backend Offline (Check port 8000)")
        st.info("Start backend via `python run.py`")

    st.markdown("---")
    st.subheader("📋 Document Metadata")
    company_name = st.text_input("Company / Ticker", value="Global Tech Dynamics (GTD)")
    doc_type = st.selectbox(
        "Document Filing Type",
        options=["10-K", "10-Q", "earnings_call", "audit_report", "credit_agreement", "press_release", "general"],
        index=0
    )

    st.markdown("---")
    st.subheader("📂 Quick-Load Samples")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Form 10-K", use_container_width=True):
            st.session_state.document_text = load_sample_file("sample_10k_risk_section.txt")
            st.rerun()
    with col2:
        if st.button("🎙️ Earnings", use_container_width=True):
            st.session_state.document_text = load_sample_file("sample_earnings_call.txt")
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ GenAI Options")
    include_summary = st.checkbox("Generate Executive Summary", value=True)
    include_mitigation = st.checkbox("Generate Mitigation Roadmap", value=True)

    st.markdown("---")
    st.subheader("📦 Export Project")
    zip_path = Path(r"C:\Users\Varun\.gemini\antigravity\scratch\financial-risk-analysis-api.zip")
    if zip_path.exists():
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Project (.zip)",
                data=f.read(),
                file_name="financial-risk-analysis-api.zip",
                mime="application/zip",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown(
        f"[📖 Swagger UI Documentation]({api_url}/docs)  \n"
        f"[📑 ReDoc Schema]({api_url}/redoc)  \n"
        f"[⬇️ Direct ZIP Download]({api_url}/download)"
    )

# ------------------------------------------------------------------------------
# Main Tabs Navigation
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 360° Risk Audit Dashboard",
    "📂 File Upload Auditor",
    "💬 Financial Risk Q&A",
    "🔬 NLP Model Playground"
])

# ==============================================================================
# TAB 1: 360° Risk Audit Dashboard
# ==============================================================================
with tab1:
    st.subheader("Document Input & Analysis")
    doc_input = st.text_area(
        "Financial Text / Item 1A Risk Factors Disclosure:",
        value=st.session_state.document_text,
        height=180,
        help="Paste financial filing excerpts, earnings call transcripts, or risk disclosures."
    )
    # Sync with session state
    st.session_state.document_text = doc_input

    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        run_audit = st.button("🚀 Run 360° Risk Audit", type="primary", use_container_width=True)
    with btn_col2:
        if st.button("🧹 Clear Input", use_container_width=False):
            st.session_state.document_text = ""
            st.session_state.last_analysis = None
            st.rerun()

    if run_audit:
        if not doc_input or len(doc_input.strip()) < 20:
            st.warning("⚠️ Please provide at least 20 characters of financial disclosure text.")
        elif not is_healthy:
            st.error("🚨 Backend API is not reachable at " + api_url + ". Make sure `python run.py` is running.")
        else:
            with st.spinner("Analyzing document with FinBERT, BERT Risk Engine, and Generative AI..."):
                payload = {
                    "text": doc_input,
                    "doc_type": doc_type,
                    "company_name": company_name,
                    "include_genai_summary": include_summary,
                    "include_mitigation": include_mitigation
                }
                try:
                    res = requests.post(f"{api_url}/api/v1/analyze/text", json=payload, timeout=40)
                    if res.status_code == 200:
                        st.session_state.last_analysis = res.json()
                        st.success("✅ Risk Analysis Complete!")
                    else:
                        st.error(f"Analysis failed ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # Render Results If Available
    if st.session_state.last_analysis:
        data = st.session_state.last_analysis
        metadata = data.get("document_metadata", {})
        finbert = data.get("finbert_sentiment", {})
        bert = data.get("bert_risk_profile", {})
        exec_summary = data.get("executive_summary")
        mitigation = data.get("mitigation_plan")

        st.markdown("---")

        # Row 1: KPI Metric Scorecards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Words / Clauses</div>
                <div class="metric-value">{metadata.get('total_words', 0):,} <span style="font-size:0.9rem; color:#94a3b8;">({metadata.get('total_sentences', 0)} sents)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            sent_label = finbert.get('dominant_sentiment', 'neutral').upper()
            sent_color = "#22c55e" if sent_label == "POSITIVE" else "#ef4444" if sent_label == "NEGATIVE" else "#94a3b8"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">FinBERT Sentiment</div>
                <div class="metric-value" style="color:{sent_color};">{sent_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            polarity = finbert.get('sentiment_polarity_index', 0.0)
            pol_color = "#22c55e" if polarity > 0 else "#ef4444" if polarity < 0 else "#94a3b8"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Polarity Index</div>
                <div class="metric-value" style="color:{pol_color};">{polarity:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            risk_lvl = bert.get('overall_risk_level', 'LOW')
            badge_class = f"badge-{risk_lvl.lower()}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Global Risk Rating</div>
                <div class="metric-value"><span class="{badge_class}">{risk_lvl}</span> <span style="font-size:1rem; color:#94a3b8;">({bert.get('overall_risk_score', 0):.2f})</span></div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Inference Speed</div>
                <div class="metric-value">{data.get('processing_time_ms', 0):.1f} <span style="font-size:0.9rem; color:#94a3b8;">ms</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Visual Charts (FinBERT Donut vs BERT Risk Bar)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("FinBERT Sentiment & Uncertainty")
            probs = finbert.get("probabilities", {})
            donut_fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[probs.get('positive', 0), probs.get('neutral', 0), probs.get('negative', 0)],
                hole=.55,
                marker_colors=['#22c55e', '#64748b', '#ef4444'],
                textinfo='label+percent'
            )])
            donut_fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=260,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(donut_fig, use_container_width=True)
            uncert = finbert.get('financial_uncertainty_score', 0.0)
            st.progress(min(1.0, uncert), text=f"Disclosure Uncertainty / Ambiguity: {uncert * 100:.1f}%")

        with chart_col2:
            st.subheader("BERT 5-Pillar Risk Breakdown")
            categories = bert.get("categories", [])
            cat_df = pd.DataFrame([
                {
                    "Category": c["category"],
                    "Risk Score": c["score"],
                    "Severity": c["risk_level"]
                }
                for c in categories
            ])
            color_map = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"}
            bar_fig = px.bar(
                cat_df,
                x="Risk Score",
                y="Category",
                orientation="h",
                color="Severity",
                color_discrete_map=color_map,
                range_x=[0, 1.0],
                text="Risk Score"
            )
            bar_fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=280,
                xaxis_title="Risk Intensity (0.0 to 1.0)",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # Row 3: Generative AI Executive Briefing
        if exec_summary:
            st.markdown("---")
            st.subheader("🧠 Generative AI Executive Briefing")
            
            # Primary Concern Callout
            st.error(f"🚨 **Primary Vulnerability:** {exec_summary.get('primary_concern')}")
            
            with st.container():
                st.markdown(f"**Executive Synthesis:**  \n{exec_summary.get('executive_summary')}")
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### ⚡ Key Risk Drivers")
                for driver in exec_summary.get("key_risk_drivers", []):
                    st.markdown(f"- {driver}")
            with col_d2:
                st.markdown("##### 🛡️ Compensating Factors & Hedges")
                for hedge in exec_summary.get("hedges_or_stabilizers", []):
                    st.markdown(f"- {hedge}")

        # Row 4: Actionable Mitigation Roadmap
        if mitigation and mitigation.get("mitigation_actions"):
            st.markdown("---")
            st.subheader("🛡️ Actionable Mitigation Roadmap")
            actions = mitigation.get("mitigation_actions", [])
            for action in actions:
                p_badge = f"badge-{action.get('priority', 'medium').lower()}"
                with st.expander(f"📌 {action.get('timeframe')}: {action.get('action_title')}", expanded=True):
                    st.markdown(f"**Priority:** <span class='{p_badge}'>{action.get('priority')}</span> | **Target Pillar:** `{action.get('target_risk_category')}`", unsafe_allow_html=True)
                    st.write(action.get("description"))

            if mitigation.get("governance_recommendations"):
                st.markdown("##### 🏛️ Governance & Board Mandates")
                for gov in mitigation.get("governance_recommendations", []):
                    st.markdown(f"- {gov}")

        # Row 5: Sentence-Level High-Risk Clause Heatmap
        flagged_sents = bert.get("top_high_risk_sentences", [])
        if flagged_sents:
            st.markdown("---")
            st.subheader("🔍 High-Risk Clause Audit & Heatmap")
            st.caption("Sentences flagged by BERT exceeding risk severity thresholds for analyst verification:")
            
            sent_data = []
            for s in flagged_sents:
                sent_data.append({
                    "Clause #": s.get("clause_index"),
                    "Category": s.get("primary_category"),
                    "Risk Score": f"{s.get('risk_score', 0):.2f}",
                    "Severity": s.get("risk_level"),
                    "Excerpt": s.get("sentence")
                })
            st.dataframe(pd.DataFrame(sent_data), use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: Document & File Uploader
# ==============================================================================
with tab2:
    st.subheader("Upload Financial Filing (.txt, .pdf, .json, .csv)")
    uploaded_file = st.file_uploader(
        "Choose a financial document:",
        type=["txt", "pdf", "json", "csv"],
        help="Upload 10-Ks, 10-Qs, audit notes, or earnings reports."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        st.info(f"📁 Selected: **{file_name}** ({len(file_bytes):,} bytes)")

        # Preview snippet
        preview_text = file_bytes.decode("utf-8", errors="replace")[:1500]
        with st.expander("📄 Document Preview (First 1500 characters)", expanded=False):
            st.text(preview_text)

        if st.button("🚀 Analyze Uploaded Document", type="primary"):
            if not is_healthy:
                st.error("Backend API is unreachable. Check port 8000.")
            else:
                with st.spinner("Processing file through document extractor and risk pipelines..."):
                    files = {"file": (file_name, file_bytes, uploaded_file.type or "application/octet-stream")}
                    form_data = {
                        "doc_type": doc_type,
                        "company_name": company_name,
                        "include_genai_summary": "true" if include_summary else "false",
                        "include_mitigation": "true" if include_mitigation else "false"
                    }
                    try:
                        res = requests.post(f"{api_url}/api/v1/analyze/file", files=files, data=form_data, timeout=60)
                        if res.status_code == 200:
                            st.session_state.last_analysis = res.json()
                            st.session_state.document_text = preview_text
                            st.success("✅ File Analysis Complete! Switch to the '360° Risk Audit Dashboard' tab to view the full visual report.")
                        else:
                            st.error(f"Analysis failed ({res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Error communicating with backend: {e}")

# ==============================================================================
# TAB 3: Financial Risk Q&A Assistant
# ==============================================================================
with tab3:
    st.subheader("💬 Context-Grounded Risk Q&A")
    st.caption("Ask questions strictly evaluated against the current document context.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        question = st.text_input(
            "Enter your question regarding the document:",
            placeholder="e.g., What debt covenants are mentioned and what is the debt maturity schedule?"
        )
    with col_q2:
        st.write("")
        st.write("")
        ask_btn = st.button("Ask Assistant", type="primary", use_container_width=True)

    # Pre-set inquiry pills
    st.markdown("**Quick Prompts:**")
    prompt_cols = st.columns(4)
    if prompt_cols[0].button("Debt Covenants & Default Risk?"):
        question = "What debt covenants and default risks are disclosed?"
        ask_btn = True
    if prompt_cols[1].button("Regulatory & Antitrust Scrutiny?"):
        question = "What regulatory inquiries or lawsuits are disclosed?"
        ask_btn = True
    if prompt_cols[2].button("Supply Chain Vulnerabilities?"):
        question = "What supply chain or operational disruptions are highlighted?"
        ask_btn = True
    if prompt_cols[3].button("FX & Currency Fluctuations?"):
        question = "What foreign exchange or interest rate exposures exist?"
        ask_btn = True

    if ask_btn and question:
        if not st.session_state.document_text:
            st.warning("Please enter or upload a document first in Tab 1 or Tab 2.")
        else:
            with st.spinner("Extracting evidence and generating grounded response..."):
                payload = {
                    "document_text": st.session_state.document_text,
                    "question": question
                }
                try:
                    res = requests.post(f"{api_url}/api/v1/generative/qa", json=payload, timeout=20)
                    if res.status_code == 200:
                        ans_data = res.json()
                        st.markdown("#### 💡 Answer")
                        st.write(ans_data.get("answer"))
                        
                        conf = ans_data.get("confidence", 0.0)
                        st.caption(f"**Attribution Confidence:** {conf * 100:.0f}%")

                        excerpts = ans_data.get("relevant_excerpts", [])
                        if excerpts:
                            with st.expander("📌 Quoted Evidentiary Clauses", expanded=True):
                                for ex in excerpts:
                                    st.markdown(f"> *\"{ex}\"*")
                    else:
                        st.error(f"Q&A failed ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==============================================================================
# TAB 4: Standalone NLP Model Playground
# ==============================================================================
with tab4:
    st.subheader("🔬 Standalone NLP Model Playground")
    st.caption("Inspect individual FinBERT sentiment probabilities or BERT risk classifications on single test clauses.")

    play_col1, play_col2 = st.columns(2)

    with play_col1:
        st.markdown("#### FinBERT Sentiment Tester")
        finbert_test_text = st.text_area(
            "Sentence to test for financial tone:",
            value="Operating profit contracted sharply due to adverse currency fluctuations and unexpected legal settlements.",
            height=100
        )
        if st.button("Evaluate FinBERT", key="fb_test"):
            try:
                r = requests.post(f"{api_url}/api/v1/finbert/sentiment", json={"text": finbert_test_text})
                if r.status_code == 200:
                    fb_res = r.json()
                    st.json(fb_res)
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Error: {e}")

    with play_col2:
        st.markdown("#### BERT 5-Pillar Risk Tester")
        bert_test_text = st.text_area(
            "Sentence to test for risk category:",
            value="A major ransomware breach shut down our enterprise order-processing systems for 48 hours.",
            height=100
        )
        if st.button("Classify Risk Category", key="bert_test"):
            try:
                r = requests.post(f"{api_url}/api/v1/bert/risk-categories", json={"text": bert_test_text})
                if r.status_code == 200:
                    b_res = r.json()
                    st.json(b_res)
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"Error: {e}")
