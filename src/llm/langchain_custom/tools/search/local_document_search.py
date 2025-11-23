from typing import Optional, List, Callable
from functools import wraps
from langchain_core.tools import tool
from pathlib import Path
from loguru import logger
from config import PATHS

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory issues


def decorator(func: Callable) -> Callable:
    def wrapper(
        base_path: Path = PATHS.llm_data / "text_documents",
    ) -> str:
        @wraps(func)
        def search_local_text_tool(
            directory_path: Optional[str] = None,
        ):
            return func(directory_path=directory_path, base_path=base_path)

        return tool(search_local_text_tool)

    return wrapper


def create_partial_tool(func: Callable, /, **fixed_kwargs) -> Callable:
    @wraps(func)
    def partial_tool(**kwargs):
        combined_kwargs = {**fixed_kwargs, **kwargs}
        return func(**combined_kwargs)

    return partial_tool


@decorator
def create_search_local_text_tool(
    directory_path: Optional[str] = None,
    base_path: Path = PATHS.llm_data / "text_documents",
):
    """
    List text files (first line preview) in a directory or a specific file
    """

    # Determine target path
    if not directory_path:
        text_documents_path = base_path
    else:
        text_documents_path = Path(directory_path)
        try:
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

    # Recursively find all text files
    try:
        text_files_list = sorted(text_documents_path.rglob("*.txt"))
    except (OSError, PermissionError) as error:
        return f"Error accessing directory {text_documents_path}: {error}"

    if not text_files_list:
        return f"No text files found in the directory: {text_documents_path}"

    # Get relative paths from the base directory
    file_paths: List[str] = []
    for txt in text_files_list:
        try:
            relative_path = txt.relative_to(text_documents_path)
            file_size = txt.stat().st_size
            size_str = (
                f"{file_size / 1024:.1f}KB"
                if file_size < 1024 * 1024
                else f"{file_size / (1024 * 1024):.1f}MB"
            )
            file_paths.append(f"{relative_path} ({size_str})")
        except (OSError, ValueError) as error:
            logger.error(f"Error processing file {txt}: {error}")
            continue

    return "\n".join(file_paths) if file_paths else "No readable text files found."


if __name__ == "__main__":
    # Test the search_local_text_documents function
    # Test with default path (empty string)
    print("Testing with default path:")
    search_tool = create_search_local_text_tool()
    result = search_tool.run({})
    print(result)

    # Test with specific path
    print("\nTesting with specific subdirectory:")
    search_tool = create_search_local_text_tool(
        base_path=PATHS.llm_data / "text_documents"
    )
    result = search_tool.run(
        {"directory_path": str(PATHS.llm_data / "text_documents" / "english")}
    )
    print(result)
