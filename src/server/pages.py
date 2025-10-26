from __future__ import annotations

from pathlib import Path
from flask import Blueprint, Response, send_from_directory, jsonify

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def serve_root_page() -> Response:
    if pages_bp.root_path:
        home_static_path = Path(pages_bp.root_path).parent / "static" / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@pages_bp.route("/home")
def serve_home_page() -> Response:
    if pages_bp.root_path:
        home_static_path = Path(pages_bp.root_path).parent / "static" / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@pages_bp.route("/csvTest")
def serve_csv_test_page() -> Response:
    if pages_bp.root_path:
        csv_test_static_path = Path(pages_bp.root_path).parent / "static" / "csvTest"
        return send_from_directory(str(csv_test_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@pages_bp.route("/image")
def serve_image_page() -> Response:
    if pages_bp.root_path:
        image_static_path = Path(pages_bp.root_path).parent / "static" / "image"
        return send_from_directory(str(image_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@pages_bp.route("/data/<path:path>")
def serve_data_files(path: str) -> Response:
    # Validate path to prevent directory traversal
    if ".." in path or path.startswith("/") or "\\" in path:
        return jsonify(
            {"error": {"code": "INVALID_PATH", "message": "Invalid file path"}}
        ), 400

    if pages_bp.root_path:
        data_static_path = Path(pages_bp.root_path).parent / "static" / "data"
        try:
            full_path = (data_static_path / path).resolve()
            if not str(full_path).startswith(str(data_static_path.resolve())):
                return jsonify(
                    {"error": {"code": "ACCESS_DENIED", "message": "Access denied"}}
                ), 403
            return send_from_directory(str(data_static_path), path)
        except (OSError, ValueError):
            return jsonify(
                {"error": {"code": "NOT_FOUND", "message": "File not found"}}
            ), 404
    raise RuntimeError("Static folder not configured")
