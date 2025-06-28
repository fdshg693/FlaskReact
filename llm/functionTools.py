from langchain_core.tools import tool
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def searchHeadwatersCompany(query: str) -> str:
    """Searches for information about headwaters Company using Tavily search engine.
    Args:
        query (str): The search query to find information about headwaters Company.
    Returns:
        str: The search results containing relevant information.
    """
    search_tool = TavilySearchResults(max_results=5)
    results = search_tool.run(query)
    return str(results)


@tool
def searchLocalDocuments(path: str = "") -> str:
    """Lists all names and summary content of text files in the specified path.
    Args:
        path (str): The path to search for text files. If empty, defaults to ../data/ directory.
                   Can be a folder path or a specific file path.
    Returns:
        str: A string containing the names and first line of each text file.
    """

    if not path:
        # Default to data directory relative to this file
        search_path = Path(__file__).parent / "../data/"
    else:
        search_path = Path(path)

    # If path doesn't exist, return error message
    if not search_path.exists():
        return f"Path does not exist: {search_path}"

    # If path is a specific file
    if search_path.is_file():
        if search_path.suffix == ".txt":
            try:
                with open(search_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    return f"{search_path.name}: {first_line}"
            except Exception as e:
                return f"Error reading file {search_path.name}: {str(e)}"
        else:
            return f"File {search_path.name} is not a text file."

    # If path is a directory, search for txt files
    try:
        txt_files = [
            f for f in search_path.iterdir() if f.suffix == ".txt" and f.is_file()
        ]
    except Exception as e:
        return f"Error accessing directory {search_path}: {str(e)}"

    if not txt_files:
        return f"No text files found in the directory: {search_path}"

    # ファイル名と１行目の内容をリストで返す
    file_info = []
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                file_info.append(f"{txt_file.name}: {first_line}")
        except Exception as e:
            file_info.append(f"{txt_file.name}: Error reading file - {str(e)}")

    return "\n".join(file_info) if file_info else "No readable text files found."


@tool
def getLocalDocuments(txtFileName: str, path: str = "") -> str:
    """Fetches text content from a specified file.
    Args:
        txtFileName (str): The name of the text file to fetch.
        Example: "oooo.txt"
        path (str): The path to search for the file. If empty, defaults to ../data/ directory.
    Returns:
        str: The content of the text file.
    """

    if not path:
        # Default to data directory relative to this file
        txt_path = Path(__file__).parent / "../data/" / txtFileName
    else:
        txt_path = Path(path) / txtFileName

    if not txt_path.exists():
        return f"File does not exist: {txt_path}"

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            docs = f.readlines()
        return "".join(docs) if docs else "No documents found."
    except Exception as e:
        return f"Error reading file {txtFileName}: {str(e)}"


if __name__ == "__main__":
    # Test the searchLocalDocuments function
    # Test with default path (empty string)
    print("Testing with default path:")
    print(searchLocalDocuments.run({"path": ""}))

    # Test with specific path
    print("\nTesting with specific data directory:")
    data_dir = str(Path(__file__).parent / "../data/")
    print(searchLocalDocuments.run({"path": data_dir}))
