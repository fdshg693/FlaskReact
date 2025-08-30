from typing import List, Dict
from langchain_core.tools import tool
from pathlib import Path
from langchain_community.tools.tavily_search import TavilySearchResults
from loguru import logger
from util.resolve_path import resolve_path

# Project paths
DATA_TEXT_DOCS_DIR = resolve_path("TEXT_DOCUMENTS_PATH")

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory issues

# In‑process read offsets for incremental chunk retrieval per file.
# Keyed by absolute Path. Value is current character offset.
_FILE_READ_OFFSETS: Dict[Path, int] = {}


@tool
def tavily_search(search_query: str) -> str:
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
def list_local_text_first_lines(limit: int = 100) -> List[Dict[str, str]]:
    """List text files under data/llm/text-documents returning file name and first line.

    Args:
        limit: Maximum number of files to include (default 100) to guard against huge directories.

    Returns:
        A list of dictionaries with keys:
            - file_name: File name (str)
            - first_line: First line content (str, may be empty if file empty)
            - skipped_reason: Present only if file skipped (e.g., too large)
    """
    results: List[Dict[str, str]] = []
    if not DATA_TEXT_DOCS_DIR.exists():
        logger.warning(
            f"Text documents directory not found: {DATA_TEXT_DOCS_DIR.as_posix()}"
        )
        return results

    count = 0
    for path in sorted(DATA_TEXT_DOCS_DIR.glob("*.txt")):
        if count >= limit:
            logger.info(
                f"Reached file limit {limit}; skipping remaining files in directory."
            )
            break

        try:
            file_size = path.stat().st_size
        except OSError as exc:  # pragma: no cover - uncommon filesystem error
            logger.error(f"Failed to stat file {path.name}: {exc}")
            continue

        if file_size > MAX_FILE_SIZE:
            logger.warning(
                f"Skipping {path.name} (size {file_size} > MAX_FILE_SIZE {MAX_FILE_SIZE})"
            )
            results.append(
                {
                    "file_name": path.name,
                    "first_line": "",
                    "skipped_reason": "file_too_large",
                }
            )
            count += 1
            continue

        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline().rstrip("\n\r")
        except OSError as exc:  # pragma: no cover - uncommon filesystem error
            logger.error(f"Failed to read file {path.name}: {exc}")
            continue

        results.append({"file_name": path.name, "first_line": first_line})
        count += 1

    logger.debug(
        f"Collected first lines for {len(results)} files from text-documents directory."
    )
    return results


@tool
def read_local_text_chunk(
    file_name: str, chunk_size: int = 500, reset: bool = False
) -> str:
    """Incrementally read a text file under the configured text-documents directory.

    Returns up to `chunk_size` characters per invocation. Subsequent calls continue
    from the previous position. If `reset` is True, reading restarts from the
    beginning. Designed for simple pagination when feeding documents to an LLM.

    Args:
        file_name: Name of the target .txt file (must reside directly under the directory).
        chunk_size: Maximum number of characters to return this call (default 500, capped at 2000).
        reset: When True, reset the internal cursor for this file before reading.

    Returns:
        A string containing the next chunk (may be empty if EOF already reached or file invalid).
    """
    # Normalize & sanitize file name (prevent path traversal)
    safe_name = Path(file_name).name
    if safe_name != file_name:
        logger.warning(f"Sanitized file name from {file_name} to {safe_name}")

    target_path = DATA_TEXT_DOCS_DIR / safe_name

    if not DATA_TEXT_DOCS_DIR.exists():
        logger.error(f"Text documents directory not found: {DATA_TEXT_DOCS_DIR}")
        return ""

    if target_path.suffix.lower() != ".txt":
        logger.warning(f"Rejected non-.txt file request: {safe_name}")
        return ""

    if not target_path.exists():
        logger.warning(f"Requested file does not exist: {safe_name}")
        return ""

    try:
        size = target_path.stat().st_size
    except OSError as exc:  # pragma: no cover
        logger.error(f"Failed to stat file {safe_name}: {exc}")
        return ""

    if size > MAX_FILE_SIZE:
        logger.warning(
            f"Skipping read: {safe_name} exceeds MAX_FILE_SIZE ({size} > {MAX_FILE_SIZE})"
        )
        return ""

    # Cap chunk size defensively
    if chunk_size <= 0:
        chunk_size = 500
    if chunk_size > 2000:  # hard safety upper bound
        chunk_size = 2000

    if reset and target_path in _FILE_READ_OFFSETS:
        _FILE_READ_OFFSETS[target_path] = 0
        logger.debug(f"Reset read offset for {safe_name}")

    # Initialize offset if first time
    offset = _FILE_READ_OFFSETS.setdefault(target_path, 0)

    # If we've already reached or passed EOF previously
    if offset >= size:
        logger.debug(f"EOF already reached for {safe_name}")
        return ""

    # Read full file into memory (utf-8). For 10MB max this is acceptable.
    try:
        text = target_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover
        logger.error(f"Failed to read file {safe_name}: {exc}")
        return ""

    # Adjust if file shrank; if grew we continue logically from prior offset
    if offset > len(text):
        logger.debug(
            f"Offset {offset} beyond current length {len(text)} for {safe_name}; resetting."
        )
        offset = 0
        _FILE_READ_OFFSETS[target_path] = 0

    end = min(offset + chunk_size, len(text))
    chunk = text[offset:end]
    _FILE_READ_OFFSETS[target_path] = end

    logger.debug(
        f"Served chunk {offset}:{end} (size {len(chunk)}) of {len(text)} for {safe_name}."
    )
    return chunk
