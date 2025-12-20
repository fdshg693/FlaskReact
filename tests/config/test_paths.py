"""Tests for src/config/paths.py module."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from config.paths import PROJECTPATHS, Paths, ensure_path_exists, find_paths, get_path


class TestPaths:
    """Test Paths class validation and behavior."""

    def test_paths_singleton_is_valid(self) -> None:
        """
        Pathsインスタンスは、正しく初期化されたこと、プロジェクトルートが存在することを確認
        """
        assert isinstance(PROJECTPATHS, Paths)
        assert isinstance(PROJECTPATHS.project_root, Path)
        assert PROJECTPATHS.project_root.exists()
        assert PROJECTPATHS.project_root.is_absolute()

    def test_all_paths_are_path_type(self) -> None:
        """
        生成されたパスがすべてPaathlibのPath型であることを確認
        """
        # Use Paths.model_fields (class) instead of PATHS.model_fields (instance)
        for field_name in Paths.model_fields:
            value = getattr(PROJECTPATHS, field_name)
            assert isinstance(value, Path), f"{field_name} is not a Path: {type(value)}"

    def test_paths_are_immutable(self) -> None:
        """
        Pathsインスタンスが不変（immutable）であることをテスト

        1. 既存の属性を変更しようとするとエラーが発生することを確認
        2. 新しい属性を追加しようとするとエラーが発生することを確認
        """
        from pydantic import ValidationError

        # Test that we cannot modify existing attributes
        with pytest.raises(ValidationError, match="frozen"):
            # プロパティに直接パスを代入して変更を試みてエラーを確認 - type: ignore[misc] は予期されるエラーを抑制
            PROJECTPATHS.project_root = Path("/tmp")  # type: ignore[misc]

        # Test that we cannot add new attributes
        with pytest.raises(ValidationError):
            # 新しい属性を追加してエラーを確認 - type: ignore[attr-defined] は予期されるエラーを抑制
            PROJECTPATHS.new_attribute = "value"  # type: ignore[attr-defined]

    def test_direct_instantiation_warns(self) -> None:
        """
        直接インスタンス化が警告を発することをテスト
        Pathsクラスは直接インスタンス化されるべきではなく、IMPORT時に自動生成されるシングルトン【PROJECTPATHS】を使用するべきである。
        直接インスタンス化しようとするとUserWarningが発生することを確認する。
        これはユーザが誤って直接インスタンス化することを防ぐためのものである。
        1. 直接インスタンス化を試み、警告が発生することを確認
        2. 発生した警告がUserWarningであることを確認
        3. 警告メッセージに "Direct instantiation" が含まれていることを確認
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = Paths(
                project_root=Path("/tmp"),
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                frontend=Path("/tmp/frontend"),
                flask_static=Path("/tmp/frontend/flask_static"),
                logs=Path("/tmp/logs"),
                original_data=Path("/tmp/data/original_sources"),
                outputs=Path("/tmp/data/outputs"),
                tmp=Path("/tmp/data/tmp"),
                ml_data=Path("/tmp/data/original_sources/ml"),
                iris_data_path=Path("/tmp/data/original_sources/ml/iris/iris.csv"),
                diabetes_data_path=Path(
                    "/tmp/data/original_sources/ml/diabetes/diabetes.csv"
                ),
                titanic_test_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_test.csv"
                ),
                titanic_train_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_train.csv"
                ),
                ml_outputs=Path("/tmp/data/outputs/ml"),
                ml_image_data=Path("/tmp/data/outputs/ml/image"),
                ml_learning_curves_dir=Path("/tmp/data/outputs/ml/learning_curves"),
                llm_data=Path("/tmp/data/original_sources/llm"),
            )

            # インスタンス生成の警告ただ一つだけが発生していることを確認
            assert len(w) == 1
            # UserWarningにより、ユーザが作成したコードが警告を発したことを確認（標準ライブラリなどの警告ではない）
            assert issubclass(w[0].category, UserWarning)
            # 警告メッセージに "Direct instantiation" が含まれていることを確認
            assert "Direct instantiation" in str(w[0].message)

    def test_path_validation_rejects_non_path(self) -> None:
        """
        パス属性にPath型以外の値を渡すとTypeErrorが発生することをテスト
        1. project_rootに整数を渡してインスタンス化を試み、TypeErrorが発生することを確認
        2. 発生したエラーのメッセージに "Expected Path or str" が含まれていることを確認
        """
        with pytest.raises(TypeError, match="Expected Path or str"):
            Paths(
                project_root=123,  # type: ignore[arg-type]
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                frontend=Path("/tmp/frontend"),
                flask_static=Path("/tmp/frontend/flask_static"),
                logs=Path("/tmp/logs"),
                original_data=Path("/tmp/data/original_sources"),
                outputs=Path("/tmp/data/outputs"),
                tmp=Path("/tmp/data/tmp"),
                ml_data=Path("/tmp/data/original_sources/ml"),
                iris_data_path=Path("/tmp/data/original_sources/ml/iris/iris.csv"),
                diabetes_data_path=Path(
                    "/tmp/data/original_sources/ml/diabetes/diabetes.csv"
                ),
                titanic_test_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_test.csv"
                ),
                titanic_train_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_train.csv"
                ),
                ml_outputs=Path("/tmp/data/outputs/ml"),
                ml_image_data=Path("/tmp/data/outputs/ml/image"),
                ml_learning_curves_dir=Path("/tmp/data/outputs/ml/learning_curves"),
                llm_data=Path("/tmp/data/original_sources/llm"),
            )

    def test_string_to_path_conversion(self) -> None:
        """
        文字列の値がPathに変換され、警告が発生することをテスト
        1. project_rootに文字列を渡してインスタンス化を試み、Pathに変換されることを確認
        2. 変換時にUserWarningが発生することを確認
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            paths = Paths(
                # Pydantic converts str -> Path at runtime; keep this as a str for test coverage.
                project_root="/tmp",  # type: ignore[arg-type]
                src=Path("/tmp/src"),
                data=Path("/tmp/data"),
                frontend=Path("/tmp/frontend"),
                flask_static=Path("/tmp/frontend/flask_static"),
                logs=Path("/tmp/logs"),
                original_data=Path("/tmp/data/original_sources"),
                outputs=Path("/tmp/data/outputs"),
                tmp=Path("/tmp/data/tmp"),
                ml_data=Path("/tmp/data/original_sources/ml"),
                iris_data_path=Path("/tmp/data/original_sources/ml/iris/iris.csv"),
                diabetes_data_path=Path(
                    "/tmp/data/original_sources/ml/diabetes/diabetes.csv"
                ),
                titanic_test_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_test.csv"
                ),
                titanic_train_data_path=Path(
                    "/tmp/data/original_sources/ml/others/titanic_train.csv"
                ),
                ml_outputs=Path("/tmp/data/outputs/ml"),
                ml_image_data=Path("/tmp/data/outputs/ml/image"),
                ml_learning_curves_dir=Path("/tmp/data/outputs/ml/learning_curves"),
                llm_data=Path("/tmp/data/original_sources/llm"),
            )

            assert isinstance(paths.project_root, Path)
            # インスタンス化の警告と、型変換の警告、合計2つの警告が発生していることを確認
            assert len(w) == 2
