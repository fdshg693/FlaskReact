from __future__ import annotations

from functools import lru_cache
from typing import List

from ml.numeric.eval_batch import evaluate_iris_batch


@lru_cache(maxsize=1)
def predict_iris_cached(
    model_path: str, scaler_path: str, features: tuple[float, ...]
) -> List[str]:
    return evaluate_iris_batch([list(features)], model_path, scaler_path)


@lru_cache(maxsize=1)
def predict_iris_batch_cached(
    model_path: str, scaler_path: str, batch: tuple[tuple[float, ...], ...]
) -> List[str]:
    return evaluate_iris_batch([list(row) for row in batch], model_path, scaler_path)
