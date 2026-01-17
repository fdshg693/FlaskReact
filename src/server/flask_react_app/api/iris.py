from __future__ import annotations

from functools import lru_cache
from typing import Annotated, List

from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from server.flask_react_app.config import get_settings
from services.iris_service import predict_iris_batch_cached, predict_iris_cached

__all__ = ["iris_bp"]

iris_bp = Blueprint("iris", __name__, url_prefix="/api")


Feature = Annotated[float, Field(ge=0, le=20)]
FeatureVector = Annotated[List[Feature], Field(min_length=4, max_length=4)]


class IrisRequest(BaseModel):
    # Explicit 4-length vector of bounded floats
    features: FeatureVector


class IrisBatchRequest(BaseModel):
    data: List[FeatureVector]


@iris_bp.route("/iris", methods=["POST"])
def handle_iris_prediction_request() -> ResponseReturnValue:
    try:
        data = request.get_json(silent=True) or {}
        # Backward compat: accept dict of 4 numbers → map to list order by sorted keys
        if isinstance(data, dict) and "features" not in data:
            # Keep deterministic order to avoid mismatch
            values: list[float] = []
            for k in sorted(data.keys()):
                values.append(float(data[k]))
            data = {"features": values}
        body = IrisRequest(**data)
    except (ValidationError, ValueError) as e:
        logger.error(f"Validation error in iris: {e}")
        return jsonify(
            {"error": {"code": "VALIDATION_ERROR", "message": "Invalid input"}}
        ), 400

    settings = get_settings()
    pred = predict_iris_cached(tuple(body.features))
    logger.info(f"Processed iris prediction for features: {body.features}")
    return jsonify({"species": pred[0]})


@iris_bp.route("/batch_iris", methods=["POST"])
def handle_user_data_batch_request() -> ResponseReturnValue:
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

    settings = get_settings()
    batch = tuple(tuple(row) for row in body.data)
    pred = predict_iris_batch_cached(batch)
    logger.info(f"Processed batch prediction for {len(body.data)} rows")
    return jsonify({"userData": pred})
