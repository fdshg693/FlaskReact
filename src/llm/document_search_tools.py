from typing import Optional, List
from langchain_core.tools import tool
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
from loguru import logger
from config import PATHS

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory issues


@tool
def tavily_search_func(search_query: str) -> str:
    tavily_search_tool = TavilySearchResults(max_results=2)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)


@tool
def search_local_text_documents(directory_path: Optional[str] = None) -> str:
    """
    List text files (first line preview) in a directory or a specific file
    """

    # Determine target path
    if not directory_path:
        text_documents_path = PATHS.llm_data / "text_documents"
    else:
        text_documents_path = Path(directory_path)
        try:
            base_path = PATHS.project_root
            if not str(text_documents_path).startswith(str(base_path)):
                logger.warning(f"Path traversal attempt detected: {directory_path}")
                return "Error: Invalid path detected for security reasons"
        except (OSError, ValueError) as e:
            logger.error(f"Path resolution error: {e}")
            return "Error: Invalid path format"

    # Fallback if path does not exist
    if not text_documents_path.exists():
        logger.info("Non-existent directory name provided")
        return f"Non-existent directory: {text_documents_path}"

    # Directory listing
    try:
        text_files_list = [
            p
            for p in text_documents_path.iterdir()
            if p.is_file() and p.suffix == ".txt"
        ]
    except (OSError, PermissionError) as error:
        return f"Error accessing directory {text_documents_path}: {error}"

    if not text_files_list:
        return f"No text files found in the directory: {text_documents_path}"

    summaries: List[str] = []
    for txt in text_files_list:
        try:
            if txt.stat().st_size > MAX_FILE_SIZE:
                summaries.append(f"{txt.name}: File too large to process")
                continue
            with txt.open("r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            summaries.append(f"{txt.name}: {first_line}")
        except (OSError, UnicodeDecodeError, IOError) as error:
            summaries.append(f"{txt.name}: Error reading file - {error}")

    return "\n".join(summaries) if summaries else "No readable text files found."


if __name__ == "__main__":
    # Test the search_local_text_documents function
    # Test with default path (empty string)
    print("Testing with default path:")
    print(search_local_text_documents.run({"directory_path": ""}))

    # Test with specific path
    print("\nTesting with specific data directory:")
    data_directory = PATHS.llm_data / "text_documents"
    print(search_local_text_documents.run({"directory_path": data_directory}))
