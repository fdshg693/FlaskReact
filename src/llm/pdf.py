from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger


def extract_text_from_pdf(pdf_path: Path | str) -> list[dict[str, Any]]:
    """
    Extract text content from a PDF file using LangChain PyPDFLoader.

    Args:
        pdf_path: Path to the PDF file to extract text from

    Returns:
        List of dictionaries containing page content and metadata

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the PDF file cannot be processed
    """
    load_dotenv("../.env")

    # Convert to Path object if string is provided
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path}")

    try:
        loader = PyPDFLoader(
            file_path=str(pdf_path),
            mode="single",
        )

        documents = loader.lazy_load()
        docs_serializable: list[dict[str, Any]] = []

        for doc in documents:
            docs_serializable.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        logger.info(f"Successfully extracted {len(docs_serializable)} pages from PDF")
        return docs_serializable

    except Exception as e:
        logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        raise ValueError(f"Could not process PDF file: {e}") from e


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent.parent / "data" / "headwaters20250521.pdf"

    logger.info("Starting PDF text extraction demonstration")

    try:
        docs = extract_text_from_pdf(pdf_path)

        # Create output filename with format: extracted_{original_filename}_{timestamp}.txt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = pdf_path.stem  # Gets filename without extension
        output_filename = f"extracted_{original_filename}_{timestamp}.txt"
        output_file = Path(__file__).parent.parent.parent / "data" / output_filename

        with output_file.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(f"{doc['page_content']}\n")

        logger.info(f"Extracted {len(docs)} documents from {pdf_path}")
        logger.info(f"Saved extracted documents to {output_file}")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"PDF extraction failed: {e}")
