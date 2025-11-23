from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_workspace(env_path: Path | None = None) -> None:
    """
    Load environment variables from the .env file located at the project root.

    Args:
        env_path: Optional path to the .env file. If None, uses PATHS.project_root / ".env".
    """
    if env_path is None:
        from config.paths import PATHS

        env_path = PATHS.project_root / ".env"

    load_dotenv(env_path)
