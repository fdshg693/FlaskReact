from __future__ import annotations

from typing import List
from llm.langchain_custom.tools.others.text_splitter import split_text


def split_text_service(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    return split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
