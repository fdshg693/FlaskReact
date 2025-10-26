from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def read_csv_from_path(
    path: str | Path, *, encoding: Optional[str] = None
) -> pd.DataFrame:
    """Read a CSV file from a filesystem path and return a pandas DataFrame.

    Uses pathlib.Path for path handling and loguru for logging to match project
    conventions.

    Args:
            path: Path or string pointing to the CSV file.
            encoding: Optional text encoding (passed to pandas.read_csv).

    Returns:
            pandas.DataFrame loaded from the CSV.

    Raises:
            FileNotFoundError: If the path does not exist or is not a file.
            pd.errors.EmptyDataError: If the CSV is empty.
            Exception: Propagates pandas parsing errors and others.
    """
    p = Path(path)
    logger.debug("read_csv_from_path() -> resolving path: {}", p)

    if not p.exists() or not p.is_file():
        logger.error("CSV file not found at path: {}", p)
        raise FileNotFoundError(f"CSV file not found: {p}")

    try:
        df = pd.read_csv(p, encoding=encoding) if encoding else pd.read_csv(p)
        logger.info("Loaded CSV with shape {} from {}", df.shape, p)
        return df
    except Exception:  # keep broad to let callers handle pandas exceptions
        logger.exception("Failed to read CSV from %s", p)
        raise


def save_csv_to_path(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    overwrite: bool = True,
    header: bool | list[str] | None = True,
    encoding: Optional[str] = None,
) -> Path:
    """Save a pandas DataFrame to CSV at the given path, creating parents.

    Args:
            df: DataFrame to save.
            path: Destination file path (string or Path).
            index: Whether to write row index.
            overwrite: If False and file exists, raise FileExistsError.
            encoding: Optional text encoding (passed to DataFrame.to_csv).

    Returns:
            Path: The absolute Path to the saved CSV file.

    Raises:
            FileExistsError: If file exists and overwrite is False.
            Exception: Propagates IO or pandas errors.
    """
    p = Path(path)
    logger.debug("save_csv_to_path() -> target path: {}", p)

    if p.exists() and not overwrite:
        logger.error("File already exists and overwrite=False: {}", p)
        raise FileExistsError(f"File exists and overwrite is False: {p}")

    # ensure parent directories exist
    if not p.parent.exists():
        logger.debug("Creating parent directories for {}", p.parent)
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Respect header and optional encoding when saving
        to_csv_kwargs = {"index": index, "header": header}
        if encoding:
            to_csv_kwargs["encoding"] = encoding
        df.to_csv(p, **to_csv_kwargs)
        logger.info("Saved CSV with shape {} to {}", getattr(df, "shape", "?"), p)
        return p.resolve()
    except Exception:
        logger.exception("Failed to save CSV to %s", p)
        raise
