from langchain_core.tools import tool
import os


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def getLocalDocuments() -> str:
    """Fetches most recent text about headwaters Company."""

    txt_path = os.path.join(
        os.path.dirname(__file__), "../data/extractedheadwaters20250523.txt"
    )

    with open(txt_path, "r") as f:
        docs = f.readlines()

    return "".join(docs) if docs else "No documents found."
