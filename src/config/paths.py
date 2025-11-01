from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Flag to control singleton instantiation
_BUILDING_SINGLETON = False


class Paths(BaseModel):
    """Immutable container for project paths.

    Use PATHS (module-level singleton) for common access:
        from config import PATHS
        data_dir = PATHS.data

    Note: Direct instantiation is prevented. Use the module-level PATHS singleton.
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

    # Prevent modification after creation
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def _ensure_path_type(cls, value: object) -> Path:
        """Ensure all fields are Path instances (not strings or other types)."""
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            logger.warning(
                "Converting string to Path (prefer passing Path objects): {}", value
            )
            return Path(value)
        msg = f"Expected Path or str, got {type(value).__name__}"
        raise TypeError(msg)

    @model_validator(mode="after")
    def _validate_paths(self) -> Paths:
        """Validate that all paths are properly formed."""
        global _BUILDING_SINGLETON  # noqa: PLW0603

        # Warn if instantiated outside singleton pattern
        if not _BUILDING_SINGLETON:
            import warnings

            warnings.warn(
                "Direct instantiation of Paths is discouraged. "
                "Use the module-level PATHS singleton instead: from config import PATHS",
                UserWarning,
                stacklevel=3,
            )

        # Additional validation: ensure project_root is absolute
        if not self.project_root.is_absolute():
            logger.warning("project_root is not absolute: {}", self.project_root)
            # Convert to absolute
            object.__setattr__(self, "project_root", self.project_root.resolve())

        # Enforce that certain fields are immediate children of project_root.
        # This ensures the canonical layout: <project_root>/src, <project_root>/data, etc.
        top_level_fields = ("src", "data", "static", "logs", "outputs", "tmp")
        for name in top_level_fields:
            value = getattr(self, name)
            # Use resolve() to normalize symlinks/relative bits
            val_parent = value.resolve().parent
            root_resolved = self.project_root.resolve()
            if val_parent != root_resolved:
                msg = (
                    f"Paths.{name} must be a direct child of project_root ({root_resolved}), "
                    f"but is: {value.resolve()}"
                )
                logger.error(msg)
                raise ValueError(msg)

        # Ensure ml_data is a direct child of data (project root -> data -> ml_data)
        if self.ml_data.resolve().parent != self.data.resolve():
            msg = (
                f"Paths.ml_data must be a direct child of data ({self.data.resolve()}), "
                f"but is: {self.ml_data.resolve()}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Ensure dataset file paths live under ml_data
        dataset_paths = (
            ("iris_data_path", self.iris_data_path),
            ("diabetes_data_path", self.diabetes_data_path),
            ("titanic_test_data_path", self.titanic_test_data_path),
            ("titanic_train_data_path", self.titanic_train_data_path),
        )
        for name, p in dataset_paths:
            try:
                if not p.resolve().is_relative_to(self.ml_data.resolve()):
                    msg = f"Paths.{name} must be located under ml_data ({self.ml_data.resolve()}), but is: {p.resolve()}"
                    logger.error(msg)
                    raise ValueError(msg)
            except AttributeError:
                # Fallback for Path implementations without is_relative_to (shouldn't occur on py>=3.9)
                p_res = p.resolve()
                ml_res = self.ml_data.resolve()
                if str(p_res).startswith(str(ml_res) + os.sep) is False:
                    msg = f"Paths.{name} must be located under ml_data ({ml_res}), but is: {p_res}"
                    logger.error(msg)
                    raise ValueError(msg)

        logger.debug("Paths instance validated successfully")
        return self

    @classmethod
    def _build(cls, root: Path | None = None) -> Paths:
        """Build a Paths instance from project root.

        INTERNAL USE ONLY - Use the module-level PATHS singleton instead.

        If root is None, it tries:
        1) env var PROJECT_ROOT
        2) two-level parent from this file (repo root)

        Args:
            root: Optional project root path. If None, auto-detected.

        Returns:
            Configured Paths instance.

        Raises:
            RuntimeError: If project root cannot be determined.
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

        if not root.exists():
            msg = f"Project root does not exist: {root}"
            logger.error(msg)
            raise RuntimeError(msg)

        # Build all paths
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

        logger.info("Building Paths instance from root: {}", root)

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
_BUILDING_SINGLETON = True
PATHS: Paths = Paths._build()
_BUILDING_SINGLETON = False


def get_path(
    *parts: str | Path, root: Path | None = None, create: bool = False
) -> Path:
    """Join parts to root (defaults to PATHS.project_root). Optionally create directories.

    Args:
        *parts: Path components to join.
        root: Base path (defaults to PATHS.project_root).
        create: If True, create the directory structure.

    Returns:
        Resolved Path object.

    Raises:
        OSError: If directory creation fails.

    Example:
        >>> p = get_path("data", "new_dir", create=True)
        >>> p = get_path(Path("outputs"), "results", root=PATHS.project_root)
    """
    base = root or PATHS.project_root
    p = base.joinpath(*[str(part) for part in parts])

    if create:
        try:
            p.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: {}", p)
        except OSError as e:
            logger.error("Failed to create directory {}: {}", p, e)
            raise

    return p


def find_paths(
    pattern: str, root: Path | None = None, recursive: bool = True
) -> list[Path]:
    """Find files via glob under root (defaults to PATHS.project_root).

    Args:
        pattern: Glob pattern (e.g., "*.py", "**/*.json").
        root: Base path to search (defaults to PATHS.project_root).
        recursive: If True, use rglob for recursive search.

    Returns:
        List of matching Path objects.

    Example:
        >>> py_files = find_paths("*.py", recursive=False)
        >>> all_json = find_paths("**/*.json")
    """
    base = root or PATHS.project_root

    try:
        if recursive:
            results = list(base.rglob(pattern))
        else:
            results = list(base.glob(pattern))
        logger.debug("Found {} paths matching '{}' in {}", len(results), pattern, base)
        return results
    except Exception as e:
        logger.error(
            "Error finding paths with pattern '{}' in {}: {}", pattern, base, e
        )
        return []


def ensure_path_exists(path: Path, *, is_file: bool = False) -> Path:
    """Ensure a path exists, creating parent directories if needed.

    Args:
        path: Path to ensure exists.
        is_file: If True, only create parent directories (not the file itself).

    Returns:
        The path (for chaining).

    Example:
        >>> log_file = ensure_path_exists(PATHS.logs / "app.log", is_file=True)
        >>> data_dir = ensure_path_exists(PATHS.data / "cache")
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured parent directory exists for file: {}", path)
    else:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: {}", path)

    return path


__all__ = ["Paths", "PATHS", "get_path", "find_paths", "ensure_path_exists"]
