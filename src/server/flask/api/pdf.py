from __future__ import annotations

import tempfile
from pathlib import Path
from flask import Blueprint, Response, jsonify, request
from loguru import logger
from pydantic import BaseModel, ValidationError

from server.config import Settings
from services.pdf_service import extract_pdf_text_service

pdf_bp = Blueprint("pdf", __name__, url_prefix="/api")


class _PdfUpload(BaseModel):
    # No fields; presence of file in form-data is validated separately.
    pass


@pdf_bp.route("/pdf", methods=["POST"])
def handle_pdf_text_extraction_request() -> Response:
    try:
        # Validate JSON body if needed later; here we just ensure request is form-data with a file
        _ = _PdfUpload()
        if "pdf" not in request.files:
            raise ValidationError.from_exception_data(
                title="No PDF file provided", line_errors=[]
            )
        pdf_file = request.files["pdf"]
        if not pdf_file:
            raise ValidationError.from_exception_data(
                title="No PDF file provided", line_errors=[]
            )
    except ValidationError as e:
        logger.error(f"Validation error in pdf: {e}")
        return jsonify(
            {"error": {"code": "VALIDATION_ERROR", "message": "No PDF file provided"}}
        ), 400

    settings = Settings()

    # Validate upload constraints
    filename = pdf_file.filename or "unnamed.pdf"
    if (
        "." not in filename
        or filename.rsplit(".", 1)[1].lower() not in settings.allowed_pdf_extensions
    ):
        return jsonify(
            {
                "error": {
                    "code": "INVALID_TYPE",
                    "message": f"File type not allowed. Allowed: {settings.allowed_pdf_extensions}",
                }
            }
        ), 400

    pdf_file.seek(0, 2)
    size = pdf_file.tell()
    pdf_file.seek(0)
    if size > settings.max_pdf_size_mb * 1024 * 1024:
        return jsonify(
            {
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"File too large. Maximum size: {settings.max_pdf_size_mb}MB",
                }
            }
        ), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file.flush()
        temp_file.seek(0)
        text = extract_pdf_text_service(Path(temp_file.name))

    return jsonify({"text": text})
