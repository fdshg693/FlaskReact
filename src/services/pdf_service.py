from __future__ import annotations

from pathlib import Path
from typing import Any

from llm.langchain_custom.tools.others.pdf import extract_text_from_pdf


def extract_pdf_text_service(path: Path) -> list[dict[str, Any]]:
    return extract_text_from_pdf(path)
