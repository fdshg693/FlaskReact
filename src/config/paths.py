from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # リポジトリのルート
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
TMP_DIR = PROJECT_ROOT / "tmp"

IRIS_DATA_PATH = DATA_DIR / "machineLearning" / "iris" / "iris.csv"
DIABETES_DATA_PATH = DATA_DIR / "machineLearning" / "diabetes" / "diabetes.csv"


def get_path(
    *parts: Union[str, Path], root: Path = PROJECT_ROOT, create: bool = False
) -> Path:
    """root に対して遅延的にパスを結合して返す。必要なら作成する。"""
    p = root.joinpath(*[str(p) for p in parts])
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def find_paths(
    pattern: str, root: Path = PROJECT_ROOT, recursive: bool = True
) -> list[Path]:
    """glob ベースで検索。呼び出し側で必要に応じて絞る。"""
    if recursive:
        return list(root.rglob(pattern))
    return list(root.glob(pattern))
