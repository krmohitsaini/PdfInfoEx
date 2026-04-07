"""
PDF to image conversion for vision-based LLM processing.

Converts each page of a PDF into a high-resolution PNG using PyMuPDF.
"""

import fitz


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """Convert every page of a PDF to a PNG byte string.

    Parameters
    ----------
    pdf_bytes : raw PDF file content.
    dpi : rendering resolution (default 200).

    Returns
    -------
    List of PNG byte strings, one per page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
    doc.close()
    return images
