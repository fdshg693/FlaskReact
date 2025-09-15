from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps
from loguru import logger

# Modern absolute imports - no sys.path manipulation needed
from server.config import Settings
from server.api.text import text_bp
from server.api.iris import iris_bp
from server.api.pdf import pdf_bp
from server.api.image import image_bp
from server.pages import pages_bp

# Settings (lazy load in factory)


# Iris prediction moved to server/api/iris.py


# No global app; use app factory below


def validate_file_upload(
    file, allowed_extensions: set[str], max_size_mb: int = 10
) -> str:
    """Validate uploaded file for security and size constraints."""
    if not file or file.filename == "":
        raise ValueError("No file selected")

    if (
        "." not in file.filename
        or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions
    ):
        raise ValueError(f"File type not allowed. Allowed: {allowed_extensions}")

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")

    return secure_filename(file.filename) or "unnamed_file"


def handle_api_errors(f):
    """Decorator to handle common API errors."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid input data"}), 400
        except FileNotFoundError as e:
            logger.error(f"File error in {f.__name__}: {e}")
            return jsonify({"error": "File processing error"}), 500
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return decorated_function


# Page routes moved to pages_bp


# API endpoint configuration


# Text API moved to text_bp


# Image analysis API
# Image/PDF API moved to blueprints


# Iris API moved to iris_bp


# Batch iris moved to iris_bp


# Data serving moved to pages_bp


def create_app() -> Flask:
    """Application factory aligning with the documented startup pattern.

    - Loads Settings() for CORS and limits
    - Registers Blueprints (pages, text, iris)
    - Adds centralized error handlers
    """
    app = Flask(__name__, static_folder="../static", static_url_path="")
    settings = Settings()

    # CORS
    CORS(
        app,
        origins=settings.cors_origins,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST"],
    )

    # Unified error handlers
    @app.errorhandler(400)
    def bad_request(e):  # type: ignore[override]
        return (
            jsonify({"error": {"code": "BAD_REQUEST", "message": str(e)}}),
            400,
        )

    @app.errorhandler(404)
    def not_found(e):  # type: ignore[override]
        return (
            jsonify({"error": {"code": "NOT_FOUND", "message": "Not found"}}),
            404,
        )

    @app.errorhandler(Exception)
    def internal_error(e):  # type: ignore[override]
        logger.exception("Unhandled error")
        return (
            jsonify(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                    }
                }
            ),
            500,
        )

    # Blueprints
    app.register_blueprint(pages_bp)
    app.register_blueprint(text_bp)
    app.register_blueprint(iris_bp)
    app.register_blueprint(pdf_bp)
    app.register_blueprint(image_bp)

    return app


if __name__ == "__main__":
    print("🚀 Starting FlaskReact application...")
    print(f"📁 Project root: {Path(__file__).parent.parent.absolute()}")
    print("🐍 Python path configured automatically")

    create_app().run(host="0.0.0.0", port=8000, debug=True)
