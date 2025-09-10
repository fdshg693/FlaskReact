from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)


class MLCompatibleDataset(Protocol):
    """Protocol for datasets compatible with this ML pipeline.

    必須要件:
    - data: 2 次元の配列ライク (n_samples, n_features)
    - target: 1 次元の配列ライク (n_samples,)

    実体は sklearn の Bunch など、.data / .target 属性を持っていれば OK。
    """

    data: npt.ArrayLike
    target: npt.ArrayLike


class _DatasetModel(BaseModel):
    """Validated dataset container conforming to MLCompatibleDataset.

    - Ensures data is a 2D numpy array (n_samples, n_features)
    - Ensures target is a 1D numpy array (n_samples,)
    - Ensures lengths match
    """

    data: np.ndarray
    target: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def _coerce_data(cls, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be 2D array-like (n_samples, n_features); got shape {arr.shape}"
            )
        return arr

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
        # When validating target, ensure length equals data's first dimension if available
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
        ds = _DatasetModel(data=data_2d, target=target_1d)
    except ValidationError as e:  # pragma: no cover - covered by specific shape checks
        # Re-raise as ValueError to keep a simpler exception surface for callers
        raise ValueError(str(e)) from e

    logger.debug(
        "Dataset created: data shape={}, target shape={}",
        ds.data.shape,
        ds.target.shape,
    )
    return ds


__all__ = ["MLCompatibleDataset", "dataframe_to_ml_dataset"]
