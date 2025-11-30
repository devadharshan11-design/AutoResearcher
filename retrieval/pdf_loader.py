import os
from typing import List, Dict

import fitz  # PyMuPDF


def load_pdf_text(pdf_path: str) -> Dict:
    """
    Load text from a PDF file using PyMuPDF (fitz).

    Returns a dictionary:
    {
        "path": pdf_path,
        "num_pages": int,
        "pages": [
            {"page_num": int, "text": str},
            ...
        ],
        "full_text": str
    }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    pages: List[Dict] = []
    full_text_parts: List[str] = []

    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")  # plain text
        text = text.strip()
        pages.append({"page_num": i + 1, "text": text})
        if text:
            full_text_parts.append(text)

    doc.close()

    full_text = "\n\n".join(full_text_parts)

    return {
        "path": pdf_path,
        "num_pages": len(pages),
        "pages": pages,
        "full_text": full_text,
    }
