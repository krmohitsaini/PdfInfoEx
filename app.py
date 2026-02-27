"""
PdfInfoEx – PDF Extraction & Testing Dashboard
================================================
Streamlit UI that orchestrates PDF detection, text/image extraction,
token comparison, and LLM-based document processing.

Two-step flow:
  1. Process  → extract text or convert to images (no API key needed)
  2. Query    → send processed content to an LLM
"""

from __future__ import annotations

import json

import streamlit as st

from config import MODEL_MAP, DEFAULT_JSON_SCHEMA_PROMPT
from core import (
    detect_document_type,
    extract_pdfminer,
    extract_pymupdf4llm,
    pdf_to_images,
    estimate_text_tokens,
    estimate_vision_tokens_for_images,
    PROVIDER_CALLERS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PdfInfoEx – PDF Extraction Dashboard",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached wrappers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_detect(pdf_bytes: bytes):
    return detect_document_type(pdf_bytes)

@st.cache_data(show_spinner="Extracting with pdfminer.six …")
def _cached_extract_pdfminer(pdf_bytes: bytes) -> str:
    return extract_pdfminer(pdf_bytes)

@st.cache_data(show_spinner="Extracting with pymupdf4llm …")
def _cached_extract_pymupdf4llm(pdf_bytes: bytes) -> str:
    return extract_pymupdf4llm(pdf_bytes)

@st.cache_data(show_spinner="Converting pages to images …")
def _cached_pdf_to_images(pdf_bytes: bytes) -> list[bytes]:
    return pdf_to_images(pdf_bytes)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("doc_type", None),
    ("avg_chars", 0.0),
    ("processed_text", None),
    ("processed_images", None),
    ("is_processed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# UI: Title
# ---------------------------------------------------------------------------
st.title("📄 PdfInfoEx – PDF Extraction Dashboard")
st.caption("Auto-detect PDF type · Compare extraction methods & tokens · Route to the right LLM")

# ---------------------------------------------------------------------------
# UI: File uploader
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Reset processing state when a new file is uploaded (or removed)
if uploaded_file is None:
    st.session_state.doc_type = None
    st.session_state.is_processed = False
    st.session_state.processed_text = None
    st.session_state.processed_images = None

# ---------------------------------------------------------------------------
# Auto-detection on upload
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    pdf_bytes: bytes = uploaded_file.getvalue()

    doc_type, avg_chars = _cached_detect(pdf_bytes)
    st.session_state.doc_type = doc_type
    st.session_state.avg_chars = avg_chars

    if doc_type == "Digital":
        st.success(
            f"✅ Detected Document Type: **Digital** "
            f"(~{int(avg_chars)} chars on page 1)"
        )
    else:
        st.info(
            f"🔍 Detected Document Type: **Scanned** "
            f"(~{int(avg_chars)} chars on page 1 — below threshold of 75)"
        )

doc_type = st.session_state.doc_type

# ---------------------------------------------------------------------------
# UI: Sidebar — extraction config + LLM provider
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Extraction method ---
    st.subheader("Extraction")
    if doc_type == "Digital":
        extraction_method = st.radio(
            "Extraction Method",
            ["pdfminer.six (raw text)", "pymupdf4llm (Markdown)"],
        )
    elif doc_type == "Scanned":
        st.radio(
            "Extraction Method",
            ["Vision Mode Enforced"],
            disabled=True,
            help="Scanned documents are sent as images to a Vision LLM.",
        )
        extraction_method = "vision"
    else:
        st.radio("Extraction Method", ["Upload a PDF first"], disabled=True)
        extraction_method = None

    st.divider()

    # --- LLM provider config ---
    st.subheader("LLM Provider")
    provider = st.selectbox("Provider", ["OpenAI", "Gemini", "Anthropic"])
    api_key = st.text_input(f"{provider} API Key", type="password")

    if doc_type:
        recommended = MODEL_MAP[provider][doc_type]
        st.selectbox("Model", [recommended], disabled=True, help="Auto-selected based on document type")
    else:
        st.selectbox("Model", ["Upload a PDF first"], disabled=True)

# ---------------------------------------------------------------------------
# UI: Token comparison (expandable)
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    with st.expander("🔢 Token Comparison (local — no API call)", expanded=False):
        if st.button("Calculate Comparative Token Usage"):
            with st.spinner("Estimating tokens …"):
                text_pdfminer = _cached_extract_pdfminer(pdf_bytes)
                text_pymupdf = _cached_extract_pymupdf4llm(pdf_bytes)
                images = _cached_pdf_to_images(pdf_bytes)

                tok_pdfminer = estimate_text_tokens(text_pdfminer)
                tok_pymupdf = estimate_text_tokens(text_pymupdf)
                tok_vision = estimate_vision_tokens_for_images(images)

            c1, c2, c3 = st.columns(3)
            c1.metric("pdfminer.six", f"{tok_pdfminer:,} tokens")
            c2.metric("pymupdf4llm", f"{tok_pymupdf:,} tokens")
            c3.metric("Vision (high-res)", f"{tok_vision:,} tokens")

            st.caption(
                "Text tokens estimated via `tiktoken` (cl100k_base). "
                "Vision tokens use OpenAI's high-res formula: 85 base + 170 × number of 512×512 tiles per page."
            )

# ---------------------------------------------------------------------------
# Step 1: Process Document (extract text / convert to images — no API key)
# ---------------------------------------------------------------------------
if uploaded_file is not None and not st.session_state.is_processed:
    st.divider()

    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if doc_type == "Digital":
            with st.spinner("Extracting text …"):
                if extraction_method and extraction_method.startswith("pdfminer"):
                    st.session_state.processed_text = _cached_extract_pdfminer(pdf_bytes)
                else:
                    st.session_state.processed_text = _cached_extract_pymupdf4llm(pdf_bytes)
            st.session_state.processed_images = None
        else:
            with st.spinner("Converting PDF to images …"):
                st.session_state.processed_images = _cached_pdf_to_images(pdf_bytes)
            st.session_state.processed_text = None

        st.session_state.is_processed = True
        st.rerun()

# ---------------------------------------------------------------------------
# Show processed content + Step 2: Query Document
# ---------------------------------------------------------------------------
if st.session_state.is_processed:
    st.divider()
    st.subheader("📄 Processed Content")

    # Show a preview of what was extracted
    if st.session_state.processed_text:
        preview = st.session_state.processed_text
        with st.expander("Extracted Text Preview", expanded=False):
            st.text_area(
                "Extracted text",
                value=preview[:3000] + ("\n\n… (truncated)" if len(preview) > 3000 else ""),
                height=250,
                disabled=True,
            )
        method_label = "Text"
    elif st.session_state.processed_images:
        num_pages = len(st.session_state.processed_images)
        st.success(f"✅ Converted {num_pages} page(s) to high-res images (Vision mode ready)")
        with st.expander("Image Preview", expanded=False):
            cols = st.columns(min(num_pages, 3))
            for i, img_bytes in enumerate(st.session_state.processed_images[:6]):
                cols[i % 3].image(img_bytes, caption=f"Page {i + 1}", use_container_width=True)
            if num_pages > 6:
                st.caption(f"… and {num_pages - 6} more page(s)")
        method_label = "Vision"

    # --- Query section (main area) ---
    st.divider()
    st.subheader("🔍 Query Document")

    query_mode = st.radio(
        "Query Mode",
        ["Default JSON Extraction", "Manual Query"],
        horizontal=True,
    )

    custom_prompt = None
    if query_mode == "Default JSON Extraction":
        st.caption(
            "Will extract: **Customer Name**, **Contract ID**, **State**, **Term** as structured JSON."
        )
    else:
        custom_prompt = st.text_area(
            "Your prompt",
            height=120,
            placeholder="Ask anything about the document…",
        )

    if st.button("🧠 Query with LLM", type="primary", use_container_width=True):
        # --- Validations ---
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
            st.stop()
        if doc_type is None:
            st.error("Document type not detected. Re-upload the PDF.")
            st.stop()

        model = MODEL_MAP[provider][doc_type]
        json_mode = query_mode == "Default JSON Extraction"
        prompt = DEFAULT_JSON_SCHEMA_PROMPT if json_mode else (custom_prompt or "Summarize this document.")

        caller = PROVIDER_CALLERS[provider]
        with st.spinner(f"Calling {provider} ({model}) …"):
            try:
                result = caller(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    text_content=st.session_state.processed_text,
                    image_bytes_list=st.session_state.processed_images,
                    json_mode=json_mode,
                )
            except Exception as exc:
                st.error(f"API error: {exc}")
                st.stop()

        # --- Display result ---
        st.subheader("📋 Extraction Result")

        try:
            parsed = json.loads(result)
            st.json(parsed)
            display_text = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            st.code(result, language="text")
            display_text = result

        with st.expander("📎 Copy-friendly output"):
            st.text_area("Result (select all → copy)", value=display_text, height=200)

elif uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
