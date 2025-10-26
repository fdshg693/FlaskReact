from __future__ import annotations

from pathlib import Path
from typing import Union
import os

from loguru import logger
from pydantic import BaseModel, ConfigDict


class Paths(BaseModel):
    """Immutable container for project paths.

    Use PATHS (module-level singleton) for common access:
        from config import PATHS
        data_dir = PATHS.data
    """

    # root
    project_root: Path

    # top-level dirs
    src: Path
    data: Path
    static: Path
    logs: Path
    outputs: Path
    tmp: Path

    # dataset dirs/files
    ml_data: Path
    iris_data_path: Path
    diabetes_data_path: Path
    titanic_test_data_path: Path
    titanic_train_data_path: Path

    # ML outputs
    ml_outputs: Path
    ml_learning_curves_dir: Path

    llm_data: Path

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @classmethod
    def build(cls, root: Path | None = None) -> "Paths":
        """Build a Paths instance from project root.

        If root is None, it tries:
        1) env var PROJECT_ROOT
        2) two-level parent from this file (repo root)
        """
        if root is None:
            env_root = os.getenv("PROJECT_ROOT")
            if env_root:
                root = Path(env_root).resolve()
                logger.debug("Using PROJECT_ROOT from env: {}", root)
            else:
                # src/config/paths.py -> parents[2] == repo root
                root = Path(__file__).resolve().parents[2]
                logger.debug("Using inferred project root: {}", root)

        src = root / "src"
        data = root / "data"
        static = root / "static"
        logs = root / "logs"
        outputs = root / "outputs"
        tmp = root / "tmp"

        ml_data = data / "machineLearning"
        iris_data_path = ml_data / "iris" / "iris.csv"
        diabetes_data_path = ml_data / "diabetes" / "diabetes.csv"
        titanic_test_data_path = ml_data / "others" / "titanic_test.csv"
        titanic_train_data_path = ml_data / "others" / "titanic_train.csv"

        ml_outputs = outputs / "machineLearning"
        ml_learning_curves_dir = ml_outputs / "learning_curves"

        llm_data = data / "llm"

        return cls(
            project_root=root,
            src=src,
            data=data,
            static=static,
            logs=logs,
            outputs=outputs,
            tmp=tmp,
            ml_data=ml_data,
            iris_data_path=iris_data_path,
            diabetes_data_path=diabetes_data_path,
            titanic_test_data_path=titanic_test_data_path,
            titanic_train_data_path=titanic_train_data_path,
            ml_outputs=ml_outputs,
            ml_learning_curves_dir=ml_learning_curves_dir,
            llm_data=llm_data,
        )


# Singleton instance to be imported across the app
PATHS: Paths = Paths.build()


def get_path(
    *parts: Union[str, Path], root: Path | None = None, create: bool = False
) -> Path:
    """Join parts to root (defaults to PATHS.project_root). Optionally create directories.

    Example:
        p = get_path("data", "new_dir", create=True)
    """
    base = root or PATHS.project_root
    p = base.joinpath(*[str(p) for p in parts])
    if create:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001 - we log and re-raise
            logger.error("Failed to create directory {}: {}", p, e)
            raise
    return p


def find_paths(
    pattern: str, root: Path | None = None, recursive: bool = True
) -> list[Path]:
    """Find files via glob under root (defaults to PATHS.project_root)."""
    base = root or PATHS.project_root
    if recursive:
        return list(base.rglob(pattern))
    return list(base.glob(pattern))


__all__ = ["Paths", "PATHS", "get_path", "find_paths"]
