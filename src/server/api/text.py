from __future__ import annotations

from flask import Blueprint, jsonify, request, Response
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

from llm.tools.others.text_splitter import split_text

text_bp = Blueprint("text", __name__, url_prefix="/api")


class TextSplitRequest(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(1000, ge=100, le=5000)
    chunk_overlap: int = Field(200, ge=0)

    @property
    def validated_overlap(self) -> int:
        # Ensure overlap < chunk_size
        return min(self.chunk_overlap, max(0, self.chunk_size - 1))


@text_bp.route("/textSplit", methods=["POST"])
def handle_text_split_request() -> Response:
    try:
        data = request.get_json(silent=True) or {}
        body = TextSplitRequest(**data)
    except ValidationError as e:
        logger.error(f"Validation error in textSplit: {e}")
        return jsonify(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input",
                    "details": e.errors(),
                }
            }
        ), 400

    chunks = split_text(
        body.text, chunk_size=body.chunk_size, chunk_overlap=body.validated_overlap
    )
    return jsonify({"chunks": chunks})
