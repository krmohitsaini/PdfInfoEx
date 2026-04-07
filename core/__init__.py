"""
core – PDF processing and LLM integration modules.

Re-exports the main public functions for convenient access:

    from core import detect_document_type, extract_pdfminer, ...
"""

from core.detector import detect_document_type
from core.extractors import extract_pdfminer, extract_pymupdf4llm
from core.image_converter import pdf_to_images
from core.token_estimator import estimate_text_tokens, estimate_vision_tokens_for_images
from core.llm_providers import PROVIDER_CALLERS

__all__ = [
    "detect_document_type",
    "extract_pdfminer",
    "extract_pymupdf4llm",
    "pdf_to_images",
    "estimate_text_tokens",
    "estimate_vision_tokens_for_images",
    "PROVIDER_CALLERS",
]
