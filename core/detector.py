"""
Auto-detection of PDF document type (Digital vs Scanned).

Uses PyMuPDF to extract text from page 1 and classifies based on
character count (threshold: 75 characters).
"""

import fitz


def detect_document_type(pdf_bytes: bytes) -> tuple[str, float]:
    """Analyse page 1 of a PDF and return (doc_type, char_count).

    Returns
    -------
    doc_type : "Digital" | "Scanned"
    char_count : number of stripped characters found on the first page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_page = doc[0]
    text = first_page.get_text()
    char_count = len(text.strip())
    doc.close()

    doc_type = "Digital" if char_count > 75 else "Scanned"
    return doc_type, float(char_count)
