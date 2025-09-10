from pathlib import Path

# core paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # リポジトリのルート
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
TMP_DIR = PROJECT_ROOT / "tmp"

# example dataset paths
IRIS_DATA_PATH = DATA_DIR / "machineLearning" / "iris" / "iris.csv"
DIABETES_DATA_PATH = DATA_DIR / "machineLearning" / "diabetes" / "diabetes.csv"

TITANIC_TEST_DATA_PATH = DATA_DIR / "machineLearning" / "others" / "titanic_test.csv"
TITANIC_TRAIN_DATA_PATH = DATA_DIR / "machineLearning" / "others" / "titanic_train.csv"

# Re-export helper functions from paths module for convenience
from .paths import get_path, find_paths  # noqa: E402,F401

# Build a concise __all__ automatically: export public globals and re-exported names.
# Keep it deterministic and explicit for commonly used symbols by filtering on UPPERCASE
# constants and callables we re-export. This avoids manual maintenance.
__all__ = [
    name
    for name, val in globals().items()
    if not name.startswith("_")
    and (name.isupper() or name in ("get_path", "find_paths"))
]
