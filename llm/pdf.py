import base64
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


def ExtractTextFromPDF(pdf_path):
    load_dotenv("../.env")
    # Pass to LLM
    llm = init_chat_model("openai:gpt-4o")
    loader = PyPDFLoader(
        file_path=pdf_path,
        # headers = None
        # password = None,
        mode="single",
    )

    documents = loader.lazy_load()
    docs_serializable = []

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
    pdf_path = os.path.join(os.path.dirname(__file__), "../data/sample.pdf")
    docs = ExtractTextFromPDF(pdf_path)
    for doc in docs:
        print(doc.page_content)
    # Expected output: Text content of the PDF file in the data/sample.pdf
