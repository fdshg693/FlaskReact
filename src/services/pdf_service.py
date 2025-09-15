from __future__ import annotations

from pathlib import Path
from llm.pdf import extract_text_from_pdf


def extract_pdf_text_service(path: Path) -> str:
    return extract_text_from_pdf(str(path))
