from __future__ import annotations

from typing import List, Annotated
from flask import Blueprint, jsonify, request, Response
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
from functools import lru_cache

from numeric.eval_batch import evaluate_iris_batch
from server.config import Settings

iris_bp = Blueprint("iris", __name__, url_prefix="/api")


Feature = Annotated[float, Field(ge=0, le=20)]
FeatureVector = Annotated[List[Feature], Field(min_length=4, max_length=4)]


class IrisRequest(BaseModel):
    # Explicit 4-length vector of bounded floats
    features: FeatureVector


class IrisBatchRequest(BaseModel):
    data: List[FeatureVector]


@lru_cache(maxsize=1)
def _predict_cached(
    model_path: str, scaler_path: str, features: tuple[float, ...]
) -> List[str]:
    return evaluate_iris_batch([list(features)], model_path, scaler_path)


@lru_cache(maxsize=1)
def _predict_batch_cached(
    model_path: str, scaler_path: str, batch: tuple[tuple[float, ...], ...]
) -> List[str]:
    return evaluate_iris_batch([list(row) for row in batch], model_path, scaler_path)


@iris_bp.route("/iris", methods=["POST"])
def handle_iris_prediction_request() -> Response:
    try:
        data = request.get_json(silent=True) or {}
        # Backward compat: accept dict of 4 numbers → map to list order by sorted keys
        if isinstance(data, dict) and "features" not in data:
            # Keep deterministic order to avoid mismatch
            values = []
            for k in sorted(data.keys()):
                values.append(float(data[k]))
            data = {"features": values}
        body = IrisRequest(**data)
    except (ValidationError, ValueError) as e:
        logger.error(f"Validation error in iris: {e}")
        return jsonify(
            {"error": {"code": "VALIDATION_ERROR", "message": "Invalid input"}}
        ), 400

    settings = Settings()
    pred = _predict_cached(
        str(settings.model_path), str(settings.scaler_path), tuple(body.features)
    )
    logger.info(f"Processed iris prediction for features: {body.features}")
    return jsonify({"species": pred[0]})


@iris_bp.route("/batch_iris", methods=["POST"])
def handle_user_data_batch_request() -> Response:
    try:
        data = request.get_json(silent=True) or {}
        body = IrisBatchRequest(**data)
    except ValidationError as e:
        logger.error(f"Validation error in batch iris: {e}")
        return jsonify(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input",
                    "details": e.errors(),
                }
            }
        ), 400

    settings = Settings()
    batch = tuple(tuple(row) for row in body.data)
    pred = _predict_batch_cached(
        str(settings.model_path), str(settings.scaler_path), batch
    )
    logger.info(f"Processed batch prediction for {len(body.data)} rows")
    return jsonify({"userData": pred})
