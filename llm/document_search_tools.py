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

from langchain_core.tools import tool
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def search_headwaters_company_info(search_query: str) -> str:
    """Searches for information about headwaters Company using Tavily search engine.
    Args:
        search_query (str): The search query to find information about headwaters Company.
    Returns:
        str: The search results containing relevant information.
    """
    tavily_search_tool = TavilySearchResults(max_results=5)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)


@tool
def search_local_text_documents(directory_path: str = "") -> str:
    """Lists all names and summary content of text files in the specified path.
    Args:
        directory_path (str): The path to search for text files. If empty, defaults to ../data/ directory.
                             Can be a folder path or a specific file path.
    Returns:
        str: A string containing the names and first line of each text file.
    """

    if not directory_path:
        # Default to data directory relative to this file
        target_search_path = Path(__file__).parent / "../data/"
    else:
        target_search_path = Path(directory_path)

    # If path doesn't exist, return error message
    if not target_search_path.exists():
        return f"Path does not exist: {target_search_path}"

    # If path is a specific file
    if target_search_path.is_file():
        if target_search_path.suffix == ".txt":
            try:
                with open(target_search_path, "r", encoding="utf-8") as file_handle:
                    first_line_content = file_handle.readline().strip()
                    return f"{target_search_path.name}: {first_line_content}"
            except Exception as error:
                return f"Error reading file {target_search_path.name}: {str(error)}"
        else:
            return f"File {target_search_path.name} is not a text file."

    # If path is a directory, search for txt files
    try:
        text_files_list = [
            file for file in target_search_path.iterdir() if file.suffix == ".txt" and file.is_file()
        ]
    except Exception as error:
        return f"Error accessing directory {target_search_path}: {str(error)}"

    if not text_files_list:
        return f"No text files found in the directory: {target_search_path}"

    # ファイル名と１行目の内容をリストで返す
    file_summary_info = []
    for text_file in text_files_list:
        try:
            with open(text_file, "r", encoding="utf-8") as file_handle:
                first_line_content = file_handle.readline().strip()
                file_summary_info.append(f"{text_file.name}: {first_line_content}")
        except Exception as error:
            file_summary_info.append(f"{text_file.name}: Error reading file - {str(error)}")

    return "\n".join(file_summary_info) if file_summary_info else "No readable text files found."


@tool
def get_local_document_content(text_file_name: str, directory_path: str = "") -> str:
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
        target_file_path = Path(__file__).parent / "../data/" / text_file_name
    else:
        target_file_path = Path(directory_path) / text_file_name

    if not target_file_path.exists():
        return f"File does not exist: {target_file_path}"

    try:
        with open(target_file_path, "r", encoding="utf-8") as file_handle:
            document_lines = file_handle.readlines()
        return "".join(document_lines) if document_lines else "No documents found."
    except Exception as error:
        return f"Error reading file {text_file_name}: {str(error)}"


if __name__ == "__main__":
    # Test the search_local_text_documents function
    # Test with default path (empty string)
    print("Testing with default path:")
    print(search_local_text_documents.run({"directory_path": ""}))

    # Test with specific path
    print("\nTesting with specific data directory:")
    data_directory = str(Path(__file__).parent / "../data/")
    print(search_local_text_documents.run({"directory_path": data_directory}))
