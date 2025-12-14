from pathlib import Path
from typing import Iterable

import pandas as pd
from data_util.csv_util import save_csv_to_path
from loguru import logger
from sklearn.datasets import load_diabetes, load_iris
from sklearn.utils._bunch import Bunch

from config import DIABETES_DATA_PATH


def _resolve_save_path(save_path: Path | str, default_filename: str) -> Path:
    """Return a concrete file path; if a directory is given, append default name.

    Args:
        save_path: Destination path or directory.
        default_filename: Filename to use when save_path is a directory.

    Returns:
        Resolved Path pointing to a file location.
    """
    dest = Path(save_path)
    if dest.is_dir():
        dest = dest / default_filename
    return dest


def _feature_names(names: Iterable[str] | None, n_cols: int) -> list[str]:
    """Normalize feature names; fallback to generic names if missing.

    Args:
        names: Provided feature names or None.
        n_cols: Number of columns in data matrix.

    Returns:
        List of column names of length n_cols.
    """
    if names:
        # Ensure it's a plain list[str]
        return [str(n) for n in names]
    return [f"feature_{i}" for i in range(n_cols)]


def _save_bunch_as_csv(
    data: Bunch,
    save_path: Path | str,
    *,
    default_filename: str,
    include_target: bool,
    index: bool,
    header: bool | list[str] | None,
) -> Path:
    """Shared implementation to persist an sklearn Bunch dataset to CSV.

    Builds a DataFrame from data.data and feature_names, optionally appends
    the target vector, and saves via save_csv_to_path.
    """
    dest = _resolve_save_path(save_path, default_filename)

    # Build DataFrame from the Bunch contents
    X = getattr(data, "data", None)
    if X is None:
        raise ValueError("Provided Bunch has no 'data' field")
    n_cols = X.shape[1] if hasattr(X, "shape") else len(X[0])  # type: ignore[index]
    names = _feature_names(getattr(data, "feature_names", None), int(n_cols))

    df = pd.DataFrame(data=X, columns=names)
    if include_target and hasattr(data, "target"):
        df["target"] = getattr(data, "target")

    saved = save_csv_to_path(df, dest, index=index, header=header)
    logger.success("Saved CSV to {}", saved)
    return saved


def save_iris_to_csv(
    save_path: Path | str,
    include_target: bool = True,
    index: bool = False,
    header: bool | list[str] | None = True,
) -> Path:
    """Load the sklearn iris dataset and save it as a CSV file.

    Args:
        save_path: Destination file path (Path or string). If a directory is provided,
            a file named "iris.csv" will be created inside it.
        include_target: Whether to include the target column in the CSV.
        index: Whether to write row names (index). Default False (preserves previous behaviour).
        header: Column names to write. True writes column names, False omits them, or a
            list of strings can be provided to use custom column names.

    Returns:
            The resolved Path to the saved CSV file.

    Notes:
            - Uses pathlib.Path for file operations and pandas for DataFrame creation.
            - Logs actions with loguru.logger.
    """
    logger.info("Loading iris dataset")
    data: Bunch = load_iris()
    return _save_bunch_as_csv(
        data,
        save_path,
        default_filename="iris.csv",
        include_target=include_target,
        index=index,
        header=header,
    )


def save_diabetes_to_csv(
    save_path: Path | str,
    include_target: bool = True,
    index: bool = False,
    header: bool | list[str] | None = True,
) -> Path:
    """Load the sklearn diabetes dataset and save it as a CSV file.

    Args:
        save_path: Destination file path (Path or string). If a directory is provided,
            a file named "diabetes.csv" will be created inside it.
        include_target: Whether to include the target column in the CSV.
        index: Whether to write row names (index). Default False.
        header: Column names to write. True writes column names, False omits them, or a
            list of strings can be provided to use custom column names.

    Returns:
        The resolved Path to the saved CSV file.
    """
    logger.info("Loading diabetes dataset")
    data: Bunch = load_diabetes()
    return _save_bunch_as_csv(
        data,
        save_path,
        default_filename="diabetes.csv",
        include_target=include_target,
        index=index,
        header=header,
    )


if __name__ == "__main__":
    # save_iris_to_csv(IRIS_DATA_PATH, include_target=True)
    save_diabetes_to_csv(DIABETES_DATA_PATH, include_target=True)
