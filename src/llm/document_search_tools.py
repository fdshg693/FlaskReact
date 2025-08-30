# LangChain Tool Collection for Document Search and Web Search
#
# This module provides custom LangChain tools for searching and retrieving information
# from multiple sources including web search and local document repositories.
#
# Key Components:
# - search_headwaters_company_info: Web search tool using Tavily API for company information
# - search_local_text_documents: File discovery tool that lists and previews text files in directories
# - get_local_document_content: Document retrieval tool that fetches complete content from text files
#
# Dependencies:
# - langchain_core.tools: Core tool decorator and functionality
# - langchain_community.tools.tavily_search: External web search integration
# - pathlib: File system path operations
#
# Usage Context:
# - AI agent function calling for information retrieval tasks
# - Document search and analysis workflows
# - Multi-source information gathering for language model applications

from typing import Optional, List
from langchain_core.tools import tool
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
from loguru import logger

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory issues


@tool
def search_headwaters_company_info(search_query: str) -> str:
    """Searches for information about headwaters Company using Tavily search engine.
    Args:
        search_query (str): The search query to find information about headwaters Company.
    Returns:
        str: The search results containing relevant information.
    """
    tavily_search_tool = TavilySearchResults(max_results=2)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)


@tool
def search_local_text_documents(directory_path: Optional[str] = None) -> str:
    """List text files (first line preview) in a directory or a specific file.

    Graceful fallback behaviour: if the provided directory_path is a simple
    invented name that does not exist we fall back to the default dataset path
    instead of failing.
    """

    # Determine target path
    if not directory_path:
        target_search_path = Path(__file__).parent.parent.parent / "data" / "ai_agent"
    else:
        target_search_path = Path(directory_path)
        try:
            resolved_path = target_search_path.resolve()
            base_path = Path(__file__).parent.parent.parent.resolve()
            if not str(resolved_path).startswith(str(base_path)):
                logger.warning(f"Path traversal attempt detected: {directory_path}")
                return "Error: Invalid path detected for security reasons"
        except (OSError, ValueError) as e:
            logger.error(f"Path resolution error: {e}")
            return "Error: Invalid path format"

    # Fallback if path does not exist
    if not target_search_path.exists():
        if (
            directory_path
            and "/" not in directory_path
            and not Path(directory_path).suffix
        ):
            logger.info(
                "Non-existent simple directory name provided; falling back to default dataset"
            )
            target_search_path = (
                Path(__file__).parent.parent.parent / "data" / "ai_agent"
            )
        else:
            logger.warning(f"Path does not exist: {target_search_path}")
            return f"Path does not exist: {target_search_path}"

    # Single file case
    if target_search_path.is_file():
        if target_search_path.suffix != ".txt":
            return f"File {target_search_path.name} is not a text file."
        try:
            if target_search_path.stat().st_size > MAX_FILE_SIZE:
                return f"File {target_search_path.name} is too large to process"
            with target_search_path.open("r", encoding="utf-8") as fh:
                first_line = fh.readline().strip()
            return f"{target_search_path.name}: {first_line}"
        except (OSError, UnicodeDecodeError, IOError) as error:
            return f"Error reading file {target_search_path.name}: {error}"

    # Directory listing
    try:
        text_files_list = [
            p
            for p in target_search_path.iterdir()
            if p.is_file() and p.suffix == ".txt"
        ]
    except (OSError, PermissionError) as error:
        return f"Error accessing directory {target_search_path}: {error}"

    if not text_files_list:
        return f"No text files found in the directory: {target_search_path}"

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


@tool
def search_headwaters_local_docs(keywords: Optional[str] = None) -> str:
    """Search default local Headwaters-related documents for keyword context.

    This specialized helper aggregates the content of all .txt files inside the
    default data/ai_agent directory that contain the word "ヘッドウォーターズ" or
    "Headwaters" (case-insensitive). Optionally additional space-separated
    keywords can be supplied.

    Usage guidance for the LLM / Agent:
      - When a user asks specifically about Headwaters apps / products and wants
        information "from local documents", prefer calling THIS tool first.
      - Do NOT pass a path; this tool always uses the default dataset.
      - Provide extra keywords only if the user supplies them (e.g., product
        names). Otherwise call with no argument.

    Args:
        keywords: Optional additional keywords (space separated) to filter lines.
    Returns:
        Up to ~30 relevant lines grouped by filename, or a message if nothing found.
    """

    base_dir = Path(__file__).parent.parent.parent / "data" / "ai_agent"
    if not base_dir.exists():  # pragma: no cover - defensive
        logger.error(f"Base document directory missing: {base_dir}")
        return f"Document directory missing: {base_dir}"

    # Prepare keyword list
    core_terms = ["ヘッドウォーターズ", "headwaters"]
    extra_terms: List[str] = []
    if keywords:
        extra_terms = [kw.strip() for kw in keywords.split() if kw.strip()]

    all_terms_lower = {t.lower() for t in core_terms + extra_terms}

    results: List[str] = []
    line_budget = 30  # Hard cap to keep responses concise

    for txt_file in sorted(base_dir.glob("*.txt")):
        try:
            if txt_file.stat().st_size > MAX_FILE_SIZE:
                logger.warning(f"Skipping large file: {txt_file.name}")
                continue
            with txt_file.open("r", encoding="utf-8") as fh:
                for line_no, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    lower_line = line.lower()
                    if any(term in lower_line for term in all_terms_lower):
                        results.append(f"{txt_file.name}:{line_no}: {line}")
                        if len(results) >= line_budget:
                            break
            if len(results) >= line_budget:
                break
        except (OSError, UnicodeDecodeError) as err:
            logger.error(f"Error reading {txt_file.name}: {err}")
            continue

    if not results:
        return "No matching Headwaters lines found in local documents."

    return "\n".join(results)


@tool
def get_local_document_content(
    text_file_name: str, directory_path: Optional[str] = None
) -> str:
    """Fetches text content from a specified file.
    Args:
        text_file_name (str): The name of the text file to fetch.
        Example: "document.txt"
        directory_path (str): The path to search for the file. If empty, defaults to ../data/ directory.
    Returns:
        str: The content of the text file.
    """

    if not directory_path:
        # Default to data directory relative to this file
        target_file_path = (
            Path(__file__).parent.parent.parent / "data" / "ai_agent" / text_file_name
        )
    else:
        directory = Path(directory_path)
        # Enhanced path validation to prevent directory traversal
        try:
            resolved_dir = directory.resolve()
            base_path = Path(__file__).parent.parent.parent.resolve()
            if not str(resolved_dir).startswith(str(base_path)):
                logger.warning(f"Path traversal attempt detected: {directory_path}")
                return "Error: Invalid path detected for security reasons"
        except (OSError, ValueError) as e:
            logger.error(f"Path resolution error: {e}")
            return "Error: Invalid path format"
        target_file_path = directory / text_file_name

    if not target_file_path.exists():
        logger.warning(f"File does not exist: {target_file_path}")
        return f"File does not exist: {target_file_path}"

    # Check file size before reading
    try:
        if target_file_path.stat().st_size > MAX_FILE_SIZE:
            logger.error(f"File too large: {target_file_path}")
            return "File too large to process"
    except OSError as e:
        logger.error(f"Error checking file size: {e}")
        return "Error accessing file"

    try:
        with open(target_file_path, "r", encoding="utf-8") as file_handle:
            document_lines = file_handle.readlines()
        return "".join(document_lines) if document_lines else "No documents found."
    except (IOError, OSError, UnicodeDecodeError) as error:
        logger.error(f"Error reading file {text_file_name}: {error}")
        return f"Error reading file {text_file_name}: {str(error)}"


if __name__ == "__main__":
    # Test the search_local_text_documents function
    # Test with default path (empty string)
    print("Testing with default path:")
    print(search_local_text_documents.run({"directory_path": ""}))

    # Test with specific path
    print("\nTesting with specific data directory:")
    data_directory = str(Path(__file__).parent.parent / "../data/ai_agent/")
    print(search_local_text_documents.run({"directory_path": data_directory}))
