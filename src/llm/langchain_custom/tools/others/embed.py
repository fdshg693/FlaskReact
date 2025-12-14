from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


def add_documents(texts: List[str]) -> None:
    """Add documents to the vector store."""
    vector_store.add_documents(
        documents=[Document(page_content=text) for text in texts]
    )


def test_vector_store():
    """Test function to add documents and perform a similarity search."""
    docs = [
        "This is a test document.",
        "Another document for testing.",
    ]
    add_documents(docs)

    results = vector_store.similarity_search_with_score(
        query="Another",
        k=5,
    )

    for result in results:
        doc, score = result
        print(f"Score: {score}, Document: {doc.page_content}")


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=3)


def test_retrieve():
    result = retriever.batch(
        [
            "test",
            "Another",
        ],
    )
    for docs in result:
        for doc in docs:
            print(f"Retrieved Document: {doc.page_content}")


if __name__ == "__main__":
    test_retrieve()
