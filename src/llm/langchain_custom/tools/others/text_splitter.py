"""
シンプルなテキスト分割ツールを提供するモジュール
RAGやドキュメント処理において、長いテキストを扱いやすいチャンクに分割するために使用されます。
"""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger

__all__ = [
    "split_text_tool",
]


def _split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Splits the input text into chunks of specified size with overlap.

    Args:
        text: The text to be split.
        chunk_size: The size of each chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size or chunk_overlap are negative.
    """
    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    logger.debug(
        f"Splitting text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)

    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


@tool
def split_text_tool(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    A simple tool function to split text into chunks.

    Args:
        text: The text to be split.
        chunk_size: The size of each chunk.
        chunk_overlap: The number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    return _split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    sample_text = (
        "This is a sample text that will be split into smaller chunks.\n"
        "The text is long enough to demonstrate the splitting functionality.\n"
        "Each chunk will have a specified size and overlap with the next chunk."
    )

    logger.info("Starting text splitting demonstration")
    chunks: list[str] = _split_text(sample_text, chunk_size=50, chunk_overlap=10)

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Chunk {i}: {chunk}")
