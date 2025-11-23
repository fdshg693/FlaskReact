"""Tests for src/config/paths.py module."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from config.paths import PATHS, Paths, ensure_path_exists, find_paths, get_path


class TestPaths:
    """Test Paths class validation and behavior."""

    def test_paths_singleton_is_valid(self) -> None:
        """Test that PATHS singleton is properly initialized."""
        assert isinstance(PATHS, Paths)
        assert isinstance(PATHS.project_root, Path)
        assert PATHS.project_root.exists()
        assert PATHS.project_root.is_absolute()

    def test_all_paths_are_path_type(self) -> None:
        """Test that all attributes are Path objects."""
        # Use Paths.model_fields (class) instead of PATHS.model_fields (instance)
        for field_name in Paths.model_fields:
            value = getattr(PATHS, field_name)
            assert isinstance(value, Path), f"{field_name} is not a Path: {type(value)}"

    def test_paths_are_immutable(self) -> None:
        """Test that Paths instance is frozen (immutable)."""
        from pydantic import ValidationError

        # Test that we cannot modify existing attributes
        with pytest.raises(ValidationError, match="frozen"):
            # Intentionally violating frozen model for testing - type: ignore[misc] suppresses expected error
            PATHS.project_root = Path("/tmp")  # type: ignore[misc]

        # Test that we cannot add new attributes
        with pytest.raises(ValidationError):
            # Intentionally adding undefined attribute for testing - type: ignore[attr-defined] suppresses expected error
            PATHS.new_attribute = "value"  # type: ignore[attr-defined]

    def test_direct_instantiation_warns(self) -> None:
        """Test that direct instantiation triggers a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Direct instantiation should warn
            # Note: paths must follow hierarchy rules (ml_data under data, datasets under ml_data)
            _ = Paths(
                project_root=Path("/tmp"),
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                static=Path("/tmp/static"),
                logs=Path("/tmp/logs"),
                outputs=Path("/tmp/outputs"),
                tmp=Path("/tmp/tmp"),
                ml_data=Path("/tmp/data/ml"),  # Must be under data
                iris_data_path=Path("/tmp/data/ml/iris.csv"),  # Must be under ml_data
                diabetes_data_path=Path("/tmp/data/ml/diabetes.csv"),
                titanic_test_data_path=Path("/tmp/data/ml/titanic_test.csv"),
                titanic_train_data_path=Path("/tmp/data/ml/titanic_train.csv"),
                ml_outputs=Path("/tmp/outputs/ml_out"),
                ml_learning_curves_dir=Path("/tmp/outputs/ml_out/curves"),
                llm_data=Path("/tmp/data/llm"),
            )

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Direct instantiation" in str(w[0].message)

    def test_path_validation_rejects_non_path(self) -> None:
        """Test that non-Path/non-str values are rejected."""
        with pytest.raises(TypeError, match="Expected Path or str"):
            Paths(
                project_root=123,  # type: ignore[arg-type]
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                static=Path("/tmp/static"),
                logs=Path("/tmp/logs"),
                outputs=Path("/tmp/outputs"),
                tmp=Path("/tmp/tmp"),
                ml_data=Path("/tmp/data/ml"),  # Must be under data
                iris_data_path=Path("/tmp/data/ml/iris.csv"),  # Must be under ml_data
                diabetes_data_path=Path("/tmp/data/ml/diabetes.csv"),
                titanic_test_data_path=Path("/tmp/data/ml/titanic_test.csv"),
                titanic_train_data_path=Path("/tmp/data/ml/titanic_train.csv"),
                ml_outputs=Path("/tmp/outputs/ml_out"),
                ml_learning_curves_dir=Path("/tmp/outputs/ml_out/curves"),
                llm_data=Path("/tmp/data/llm"),
            )

    def test_string_to_path_conversion(self) -> None:
        """Test that string values are converted to Path with warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            paths = Paths(
                project_root="/tmp",  # String instead of Path
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                static=Path("/tmp/static"),
                logs=Path("/tmp/logs"),
                outputs=Path("/tmp/outputs"),
                tmp=Path("/tmp/tmp"),
                ml_data=Path("/tmp/data/ml"),  # Must be under data
                iris_data_path=Path("/tmp/data/ml/iris.csv"),  # Must be under ml_data
                diabetes_data_path=Path("/tmp/data/ml/diabetes.csv"),
                titanic_test_data_path=Path("/tmp/data/ml/titanic_test.csv"),
                titanic_train_data_path=Path("/tmp/data/ml/titanic_train.csv"),
                ml_outputs=Path("/tmp/outputs/ml_out"),
                ml_learning_curves_dir=Path("/tmp/outputs/ml_out/curves"),
                llm_data=Path("/tmp/data/llm"),
            )

            assert isinstance(paths.project_root, Path)
            # Should have warnings for string conversion and direct instantiation
            assert len(w) >= 1


class TestHelperFunctions:
    """Test helper functions in paths module."""

    def test_get_path_joins_correctly(self) -> None:
        """Test get_path joins path components correctly."""
        result = get_path("data", "test")
        expected = PATHS.project_root / "data" / "test"
        assert result == expected

    def test_get_path_with_custom_root(self) -> None:
        """Test get_path with custom root."""
        custom_root = Path("/tmp")
        result = get_path("sub", "dir", root=custom_root)
        assert result == custom_root / "sub" / "dir"

    def test_get_path_creates_directory(self, tmp_path: Path) -> None:
        """Test get_path creates directories when requested."""
        test_dir = get_path("test_dir", root=tmp_path, create=True)
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_find_paths_recursive(self) -> None:
        """Test find_paths with recursive search."""
        py_files = find_paths("*.py", root=PATHS.src, recursive=True)
        assert len(py_files) > 0
        assert all(p.suffix == ".py" for p in py_files)

    def test_find_paths_non_recursive(self) -> None:
        """Test find_paths with non-recursive search."""
        # Search only in project root (non-recursive)
        md_files = find_paths("*.md", root=PATHS.project_root, recursive=False)
        # Should find README.md in root but not in subdirectories
        assert any(f.name == "README.md" for f in md_files)

    def test_ensure_path_exists_directory(self, tmp_path: Path) -> None:
        """Test ensure_path_exists creates directory."""
        test_dir = tmp_path / "new_dir"
        result = ensure_path_exists(test_dir)
        assert result == test_dir
        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_ensure_path_exists_file_parents(self, tmp_path: Path) -> None:
        """Test ensure_path_exists creates parent directories for file."""
        test_file = tmp_path / "sub" / "dir" / "file.txt"
        result = ensure_path_exists(test_file, is_file=True)
        assert result == test_file
        assert test_file.parent.exists()
        assert not test_file.exists()  # File itself should not be created
