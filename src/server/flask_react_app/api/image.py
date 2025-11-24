from __future__ import annotations

from flask import Blueprint, Response, jsonify, request
from loguru import logger
from pydantic import BaseModel, ValidationError
import torch

from server.flask_react_app.config import Settings
from services.image_service import predict_image_service
from util.filestorage_to_tensor import filestorage_to_tensor_no_tv

__all__ = ["image_bp"]

image_bp = Blueprint("image", __name__, url_prefix="/api")


class _ImageUpload(BaseModel):
    pass


@image_bp.route("/image", methods=["POST"])
def handle_image_analysis_request() -> Response:
    try:
        _ = _ImageUpload()
        if "image" not in request.files:
            raise ValidationError.from_exception_data(
                title="No image file provided", line_errors=[]
            )
        image_file = request.files["image"]
        if not image_file:
            raise ValidationError.from_exception_data(
                title="No image file provided", line_errors=[]
            )
    except ValidationError as e:
        logger.error(f"Validation error in image: {e}")
        return jsonify(
            {"error": {"code": "VALIDATION_ERROR", "message": "No image file provided"}}
        ), 400

    settings = Settings()

    filename = image_file.filename or "unnamed"
    if (
        "." not in filename
        or filename.rsplit(".", 1)[1].lower() not in settings.allowed_image_extensions
    ):
        return jsonify(
            {
                "error": {
                    "code": "INVALID_TYPE",
                    "message": f"File type not allowed. Allowed: {settings.allowed_image_extensions}",
                }
            }
        ), 400

    image_file.seek(0, 2)
    size = image_file.tell()
    image_file.seek(0)
    if size > settings.max_image_size_mb * 1024 * 1024:
        return jsonify(
            {
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"File too large. Maximum size: {settings.max_image_size_mb}MB",
                }
            }
        ), 400

    img_tensor: torch.Tensor = filestorage_to_tensor_no_tv(image_file)

    predict_result = predict_image_service(
        checkpoint_path=str(settings.checkpoint_path), img_tensor=img_tensor
    )
    return jsonify({"description": predict_result})
