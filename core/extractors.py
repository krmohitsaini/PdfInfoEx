"""
Text extraction from PDF files.

Provides two methods:
  - pdfminer.six  → raw plain text
  - pymupdf4llm   → Markdown-formatted text
"""

import io
import tempfile
from pathlib import Path


def extract_pdfminer(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF using pdfminer.six."""
    from pdfminer.high_level import extract_text
    return extract_text(io.BytesIO(pdf_bytes))


def extract_pymupdf4llm(pdf_bytes: bytes) -> str:
    """Extract Markdown-formatted text from a PDF using pymupdf4llm."""
    import pymupdf4llm

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        md_text = pymupdf4llm.to_markdown(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    return md_text
