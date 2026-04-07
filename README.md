# PdfInfoEx – PDF Extraction & Testing Dashboard

A Streamlit-powered dashboard that auto-detects PDF types (Digital vs Scanned), compares extraction methods and token costs, and routes documents to the right LLM for structured data extraction.

## Features

- **Auto-Detection** — Classifies PDFs as Digital or Scanned using PyMuPDF character analysis on page 1 (threshold: 75 chars).
- **Dual Extraction** — Choose between `pdfminer.six` (raw text) or `pymupdf4llm` (Markdown) for digital PDFs.
- **Vision Mode** — Scanned PDFs are automatically converted to high-res images and sent to vision-capable LLMs.
- **Multi-Provider Support** — Route to OpenAI, Gemini, Anthropic, or a locally running LM Studio model:
  | Provider   | Digital (lightweight)       | Scanned (vision)            |
  |------------|-----------------------------|-----------------------------|
  | OpenAI     | gpt-4o-mini                 | gpt-4o                      |
  | Gemini     | gemini-1.5-flash            | gemini-1.5-pro              |
  | Anthropic  | claude-3-5-haiku-latest     | claude-3-5-sonnet-latest    |
  | LM Studio  | any loaded model            | vision-capable model only   |
- **Local LLM via LM Studio** — Run inference entirely on-device. No API key required; just enter the LM Studio server URL (`http://localhost:1234/v1` by default) and the name of the model you have loaded.
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
│   └── llm_providers.py    # API routing (OpenAI, Gemini, Anthropic, LM Studio)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and dependencies automatically.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone / navigate to the project
cd PdfInfoEx

# Install dependencies and create the virtual environment
uv sync
```

### Using pip

```bash
cd PdfInfoEx

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Usage

### Using uv

```bash
uv run streamlit run app.py
```

### Using pip

```bash
streamlit run app.py
```

1. **Upload** a PDF file.
2. The app auto-detects the document type and displays a banner.
3. **Configure** the sidebar — pick a provider, paste your API key (or enter the LM Studio server URL for local inference), choose extraction method and query mode.
4. (Optional) Expand **Token Comparison** to see estimated token costs across all three methods.
5. Click **Process Document** to run the extraction via your chosen LLM.
6. Copy the result from the JSON viewer or the copy-friendly output area.

## Using LM Studio (local inference)

1. Download and open [LM Studio](https://lmstudio.ai).
2. Download a model (e.g. `llama-3.2-3b-instruct`). For scanned PDFs, download a vision model (e.g. `llava`).
3. Go to the **Developer** tab and click **Start Server** (default port: `1234`).
4. In the app sidebar, select **LM Studio** as the provider.
5. The **Server URL** field defaults to `http://localhost:1234/v1` — change it only if you use a different port.
6. Enter the model identifier shown in LM Studio and proceed as normal.

> No API key is required for local inference.

## Requirements

- Python 3.14+
- Dependencies are declared in [pyproject.toml](pyproject.toml) (used by uv) and mirrored in [requirements.txt](requirements.txt) (for pip).
