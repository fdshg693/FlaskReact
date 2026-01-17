from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.numeric.dataset import MLDatasetConverter


def test_dataframe_to_ml_dataset_happy_path():
    df: pd.DataFrame = pd.DataFrame(
        {
            "data": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            "target": [0, 1, 0],
        }
    )

    ds = MLDatasetConverter.convert(source=df, dtype=np.dtype("float64"))

    assert hasattr(ds, "data") and hasattr(ds, "target")
    assert isinstance(ds.data, np.ndarray) and isinstance(ds.target, np.ndarray)
    assert ds.data.shape == (3, 3)
    assert ds.target.shape == (3,)
    np.testing.assert_array_equal(ds.target, np.array([0, 1, 0]))


def test_missing_columns_raises():
    df = pd.DataFrame({"data": [[1, 2], [3, 4]]})  # missing target
    with pytest.raises(ValueError):
        MLDatasetConverter.convert(df)


def test_inconsistent_row_lengths_raises():
    df = pd.DataFrame(
        {
            "data": [
                [1.0, 2.0],
                [3.0, 4.0, 5.0],  # different length
            ],
            "target": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        MLDatasetConverter.convert(df)


def test_length_mismatch_raises():
    # Construct the mismatched DataFrame inside the raises context because
    # pandas itself raises on differing column lengths.
    with pytest.raises(ValueError):
        df = pd.DataFrame(
            {
                "data": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                "target": [0],  # mismatch
            }
        )
        MLDatasetConverter.convert(df)
