from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


def extract_text_from_pdf(pdf_path: Path | str) -> List[Dict[str, Any]]:
    """
    Extract text content from a PDF file using LangChain PyPDFLoader.

    Args:
        pdf_path: Path to the PDF file to extract text from

    Returns:
        List of dictionaries containing page content and metadata
    """
    load_dotenv("../.env")

    # Convert to Path object if string is provided
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)

    loader = PyPDFLoader(
        file_path=str(pdf_path),
        mode="single",
    )

    documents = loader.lazy_load()
    docs_serializable: List[Dict[str, Any]] = []

    for doc in documents:
        docs_serializable.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return docs_serializable


if __name__ == "__main__":
    # Fetch PDF data from local file
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "../data/headwaters20250521.pdf"
    docs = extract_text_from_pdf(pdf_path)

    # Save the documents to a file
    output_file = current_dir / "../data/extracted_docs.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(f"{doc['page_content']}\n")

    print(f"Extracted {len(docs)} documents from {pdf_path}.")
    print(f"Saved extracted documents to {output_file}.")
