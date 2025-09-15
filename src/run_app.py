"""
Project launcher for FlaskReact.

Always start the app with:

    uv run run_app.py

This ensures proper module imports and path setup per repo guidance.
"""

from __future__ import annotations

from pathlib import Path
from loguru import logger

# Import the Flask app factory without side effects
from server.app import create_app


def main() -> None:
    project_root = Path(__file__).resolve().parent
    logger.info(f"Starting FlaskReact from {project_root}")

    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
