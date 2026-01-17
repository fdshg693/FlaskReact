from __future__ import annotations

from functools import lru_cache
from typing import List

from config import PROJECTPATHS
from ml.numeric.eval_batch import evaluate_iris_batch


@lru_cache(maxsize=1)
def predict_iris_cached(features: tuple[float, ...]) -> List[str]:
    return evaluate_iris_batch(
        [list(features)],
        PROJECTPATHS.default_iris_model_path,
        PROJECTPATHS.default_iris_scaler_path,
    )


@lru_cache(maxsize=1)
def predict_iris_batch_cached(batch: tuple[tuple[float, ...], ...]) -> List[str]:
    return evaluate_iris_batch(
        [list(row) for row in batch],
        PROJECTPATHS.default_iris_model_path,
        PROJECTPATHS.default_iris_scaler_path,
    )
