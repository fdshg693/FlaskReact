from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pathlib import Path
from data_util.csv_util import read_csv_from_path


class MLCompatibleDataset(BaseModel):
    """Concrete dataset type compatible with execute_machine_learning_pipeline.

    Required properties:
    - data: 2D numpy array (n_samples, n_features)
    - target: 1D numpy array (n_samples,)

    Optional metadata (not used by the pipeline but useful for logging/UX):
    - feature_names: list of feature names
    - target_names: list of target/class names
    - descr: free-form description
    """

    data: np.ndarray
    target: np.ndarray
    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    descr: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ----------------------- validators & coercion -----------------------
    @field_validator("data", mode="before")
    @classmethod
    def _coerce_data(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be 2D array-like (n_samples, n_features); got shape {arr.shape}"
            )
        # Prefer float32 for downstream torch conversion
        return arr.astype(np.float32, copy=False)

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v).reshape(-1)
        if arr.ndim != 1:
            raise ValueError("target must be 1D array-like (n_samples,)")
        return arr

    @field_validator("target")
    @classmethod
    def _match_lengths(cls, t: np.ndarray, info: ValidationInfo) -> np.ndarray:  # type: ignore[override]
        # Ensure length equals number of rows in data
        data = info.data.get("data") if hasattr(info, "data") else None
        if (
            isinstance(data, np.ndarray)
            and data.ndim == 2
            and data.shape[0] != t.shape[0]
        ):
            raise ValueError(
                f"Length mismatch: len(target)={t.shape[0]} vs n_samples (data)={data.shape[0]}"
            )
        return t


def _stack_data_column(
    col: Iterable[Any], *, dtype: np.dtype | None = np.float32
) -> np.ndarray:
    """Stack an iterable of 1D feature vectors into a 2D array.

    Each element in col must be 1D array-like of the same length.
    """
    rows: list[np.ndarray] = []
    for i, item in enumerate(col):
        arr = np.asarray(item)
        if arr.ndim != 1:
            raise ValueError(f"Row {i}: data element must be 1D; got shape {arr.shape}")
        rows.append(arr)
    try:
        stacked = np.stack(rows, axis=0)
    except Exception as exc:  # pragma: no cover - numpy provides clear error details
        raise ValueError("Failed to stack 'data' column into 2D array") from exc
    if dtype is not None:
        stacked = stacked.astype(dtype, copy=False)
    return stacked


def dataframe_to_ml_dataset(
    df: pd.DataFrame,
    *,
    data_col: str = "data",
    target_col: str = "target",
    dtype: np.dtype | None = np.float32,
) -> MLCompatibleDataset:
    """Convert a DataFrame with 'data' and 'target' columns to MLCompatibleDataset.

    Assumptions:
    - df[data_col]: each cell is a 1D array-like of features with consistent length
    - df[target_col]: scalar label/value per row

    Returns an object exposing .data and .target attributes.
    Raises ValueError on validation failure.
    """

    missing: list[str] = [c for c in (data_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.debug(
        "Converting DataFrame to ML dataset with columns data_col='{}', target_col='{}' (n_rows={})",
        data_col,
        target_col,
        len(df),
    )

    data_2d = _stack_data_column(df[data_col].to_list(), dtype=dtype)
    target_1d = np.asarray(df[target_col].to_numpy()).reshape(-1)

    try:
        ds = MLCompatibleDataset(data=data_2d, target=target_1d)
    except ValidationError as e:  # pragma: no cover - covered by specific shape checks
        # Re-raise as ValueError to keep a simpler exception surface for callers
        raise ValueError(str(e)) from e

    logger.debug(
        "Dataset created: data shape={}, target shape={}",
        ds.data.shape,
        ds.target.shape,
    )
    return ds


def from_sklearn_bunch(bunch: Any) -> MLCompatibleDataset:
    """Create MLCompatibleDataset from an sklearn-style Bunch object.

    Accepts objects exposing attributes `.data` and `.target` and optionally
    `.feature_names`, `.target_names`, `.DESCR`.
    """
    try:
        data = getattr(bunch, "data")
        target = getattr(bunch, "target")
    except AttributeError as exc:  # pragma: no cover - trivial attr access
        raise ValueError(
            "Provided object lacks 'data' and/or 'target' attributes"
        ) from exc

    feature_names = getattr(bunch, "feature_names", None)
    target_names = getattr(bunch, "target_names", None)
    descr = getattr(bunch, "DESCR", None)

    try:
        return MLCompatibleDataset(
            data=data,
            target=target,
            feature_names=list(feature_names) if feature_names is not None else None,
            target_names=list(target_names) if target_names is not None else None,
            descr=str(descr) if descr is not None else None,
        )
    except ValidationError as e:  # pragma: no cover
        raise ValueError(str(e)) from e


def ml_dataset_from_csv(
    path: str | Path,
    *,
    features: list[str],
    target: str | None = None,
    encoding: str | None = None,
    dropna: bool = False,
    dtype: np.dtype | None = np.float32,
) -> MLCompatibleDataset:
    """Load a CSV and convert into MLCompatibleDataset.

    Contract:
    - Selects given feature columns (and optional target) from the CSV.
    - Coerces features to a 2D float array (dtype arg, default float32).
    - Validates length consistency between X and y when target is provided.

    Args:
        path: CSV file path.
        features: List of feature column names to extract from the CSV.
        target: Optional target column name. If None, y will be an empty array of len 0.
        encoding: Optional text encoding for CSV reading.
        dropna: If True, drop rows with NA across selected columns.
        dtype: Desired dtype for feature matrix.

    Returns:
        MLCompatibleDataset

    Raises:
        FileNotFoundError: If CSV path is invalid.
        ValueError: If required columns are missing or stacking fails.
        pandas errors propagated from CSV parsing.
    """
    df = read_csv_from_path(path, encoding=encoding)

    missing_cols = [c for c in features if c not in df.columns]
    if target is not None and target not in df.columns:
        missing_cols.append(target)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    sel_cols = list(features) + ([target] if target is not None else [])
    work = df.loc[:, sel_cols].copy()

    if dropna:
        before = len(work)
        work = work.dropna(axis=0, how="any")
        after = len(work)
        if before != after:
            logger.info(
                "Dropped {} rows due to NA across selected columns", before - after
            )

    # Build X (2D) and optional y (1D)
    try:
        X = np.asarray(work[features].to_numpy())
        if X.ndim != 2:
            raise ValueError(f"Features matrix must be 2D; got {X.ndim}D")
        if dtype is not None:
            X = X.astype(dtype, copy=False)
    except Exception as exc:
        raise ValueError("Failed to build feature matrix from CSV") from exc

    if target is not None:
        y = np.asarray(work[target].to_numpy()).reshape(-1)
        ds = MLCompatibleDataset(data=X, target=y)
    else:
        # no target; create empty y with length 0 to satisfy model, or raise
        y = np.empty((0,), dtype=np.float32)
        ds = MLCompatibleDataset(data=X, target=y)

    logger.debug(
        "ml_dataset_from_csv -> data shape={}, target shape={}",
        ds.data.shape,
        ds.target.shape,
    )
    return ds


__all__ = [
    "MLCompatibleDataset",
    "dataframe_to_ml_dataset",
    "from_sklearn_bunch",
    "ml_dataset_from_csv",
]
