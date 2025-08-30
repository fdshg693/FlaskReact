#!/usr/bin/env python3
"""
FlaskReact application launcher.

This script sets up the proper Python path and launches the Flask application
with the correct environment configuration.
"""

import os
import sys
from pathlib import Path


def setup_environment() -> None:
    """Setup the Python environment for the FlaskReact application."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.absolute()

    # Add project root to Python path if not already present
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if project_root_str not in current_pythonpath:
        if current_pythonpath:
            os.environ["PYTHONPATH"] = f"{project_root_str}:{current_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = project_root_str


def main() -> None:
    """Main entry point for the application."""
    setup_environment()

    # Import and run the Flask app
    from server.app import app

    print("🚀 Starting FlaskReact application...")
    print(f"📁 Project root: {Path(__file__).parent.parent.absolute()}")
    print("🐍 Python path configured automatically")

    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
