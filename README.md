# PdfInfoEx – PDF Extraction & Testing Dashboard

A Streamlit-powered dashboard that auto-detects PDF types (Digital vs Scanned), compares extraction methods and token costs, and routes documents to the right LLM for structured data extraction.

## Features

- **Auto-Detection** — Classifies PDFs as Digital or Scanned using PyMuPDF character analysis on page 1 (threshold: 75 chars).
- **Dual Extraction** — Choose between `pdfminer.six` (raw text) or `pymupdf4llm` (Markdown) for digital PDFs.
- **Vision Mode** — Scanned PDFs are automatically converted to high-res images and sent to vision-capable LLMs.
- **Multi-Provider Support** — Route to OpenAI, Gemini, or Anthropic with automatic model selection:
  | Provider  | Digital (lightweight)       | Scanned (vision)            |
  |-----------|-----------------------------|-----------------------------|
  | OpenAI    | gpt-4o-mini                 | gpt-4o                      |
  | Gemini    | gemini-1.5-flash            | gemini-1.5-pro              |
  | Anthropic | claude-3-5-haiku-latest     | claude-3-5-sonnet-latest    |
- **Token Comparison** — Local estimation (no API calls) comparing pdfminer, pymupdf4llm, and vision token costs side-by-side.
- **Structured Output** — Default JSON extraction mode pulls Customer Name, Contract ID, State, and Term. Switch to Manual Query for free-form prompts.

## Project Structure

```
PdfInfoEx/
├── app.py                  # Streamlit UI orchestrator
├── config.py               # Constants (MODEL_MAP, default prompts)
├── core/
│   ├── __init__.py         # Re-exports all public functions
│   ├── detector.py         # PDF type auto-detection (Digital vs Scanned)
│   ├── extractors.py       # Text extraction (pdfminer.six, pymupdf4llm)
│   ├── image_converter.py  # PDF → PNG conversion for vision mode
│   ├── token_estimator.py  # Token counting (tiktoken + vision tile formula)
│   └── llm_providers.py    # API routing (OpenAI, Gemini, Anthropic)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone / navigate to the project
cd PdfInfoEx

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

1. **Upload** a PDF file.
2. The app auto-detects the document type and displays a banner.
3. **Configure** the sidebar — pick a provider, paste your API key, choose extraction method and query mode.
4. (Optional) Expand **Token Comparison** to see estimated token costs across all three methods.
5. Click **Process Document** to run the extraction via your chosen LLM.
6. Copy the result from the JSON viewer or the copy-friendly output area.

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for the full dependency list.
