"""
Minimal, robust text extraction for DOCX and PDF.
Keep it pure functions so it’s easy to test and replace later (e.g., OCR).
"""
import os
from typing import Tuple
from unidecode import unidecode


def _read_docx(path: str) -> str:
    """
    Extract text from a .docx using python-docx.
    Paragraph-join keeps section boundaries simple for Step 1.
    """
    import docx  # imported here to avoid import cost if unused
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def _read_pdf(path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Page-by-page concatenation; layout-aware improvements can be added later.
    """
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)


def extract_text(path: str) -> Tuple[str, int]:
    """
    Unified entrypoint. Returns (text, pages) where pages is reserved for later.
    Unknown extensions are read as plain text.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        txt, pages = _read_docx(path), None
    elif ext == ".pdf":
        txt, pages = _read_pdf(path), None
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        pages = None
    # Normalize for downstream tokenization; removes odd diacritics/glyphs.
    return unidecode(txt), pages
