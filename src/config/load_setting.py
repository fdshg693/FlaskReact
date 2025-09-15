def load_dotenv_workspace():
    """Load environment variables from the .env file located at the project root."""
    from dotenv import load_dotenv
    from pathlib import Path
    from config import PATHS

    env_path: Path = PATHS.project_root / ".env"
    load_dotenv(env_path)
