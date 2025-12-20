"""Flask API blueprints for various endpoints."""

from __future__ import annotations

from .image import image_bp
from .iris import iris_bp
from .pdf import pdf_bp
from .text import text_bp

__all__ = ["image_bp", "iris_bp", "pdf_bp", "text_bp"]
