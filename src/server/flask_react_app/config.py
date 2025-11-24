from __future__ import annotations

from pathlib import Path
from typing import List, Set
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

from config import PATHS


class Settings(BaseSettings):
    """Flask アプリケーションの設定を管理するクラス.

    環境変数または .env ファイルから設定値を読み込みます。
    すべての環境変数は 'FLASKREACT_' プレフィックスを持つ必要があります。

    Attributes:
        cors_origins: CORS で許可するオリジンのリスト
        allowed_image_extensions: アップロード可能な画像の拡張子セット
        allowed_pdf_extensions: アップロード可能な PDF の拡張子セット
        max_image_size_mb: 画像ファイルの最大サイズ (MB)
        max_pdf_size_mb: PDF ファイルの最大サイズ (MB)
        app_root: アプリケーションのルートディレクトリパス
        model_path: 機械学習モデルのファイルパス
        scaler_path: データスケーラーのファイルパス
        checkpoint_path: モデルチェックポイントのファイルパス
    """

    # SettingsConfigDict を使用して設定を指定
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FLASKREACT_",
        extra="ignore",
    )
    # CORS設定: Reactフロントエンドからのリクエストを許可
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # Upload できるファイルの限界値を設定
    allowed_image_extensions: Set[str] = Field(
        default_factory=lambda: {"png", "jpg", "jpeg", "gif"}
    )
    allowed_pdf_extensions: Set[str] = Field(default_factory=lambda: {"pdf"})
    max_image_size_mb: int = 5
    max_pdf_size_mb: int = 10

    # TODO: 起動時のパスのファイルが存在するかをチェックするロジックを追加する必要あり
    # TODO: ファイルが存在しない場合のデフォルトの動作を定義する必要あり
    # TODO: パスの記述方法をべた書きから変更する必要あり（修正案：①環境変数で指定可能にする,②最新のモデルを自動検出する関数を作成,③設定ファイル(YAML/JSON)で管理）
    # モデルのファイルパス設定
    app_root: Path = Field(default_factory=lambda: PATHS.src)
    model_path: Path = Field(
        default_factory=lambda: PATHS.ml_outputs
        / "param"
        / "models_20250712_021710.pth_validation"  # 検証用に`_validation`を追加中
    )
    scaler_path: Path = Field(
        default_factory=lambda: PATHS.ml_outputs
        / "scaler"
        / "scaler.joblib_validation"  # 検証用に`_validation`を追加中
    )
    checkpoint_path: Path = Field(
        default_factory=lambda: PATHS.ml_outputs
        / "checkpoints"
        / "2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset"
        / "best_accuracy.pth_validation"  # 検証用に`_validation`を追加中
    )

    # `@field_validator` 引数のフィールドに値がセットされる際に自動的に検証メソッドが実行(pydantic v2)
    @field_validator("model_path", "scaler_path", "checkpoint_path")
    # `@classmethod` クラスメソッドとして定義することを明記、第一引数にクラス自身を受け取る、インスタンス化せずに呼び出せる
    @classmethod
    def _validate_path_existence(cls, value: Path, info) -> Path:
        """各パスフィールドの存在を確認し、存在しない場合は警告ログを出力.

        Args:
            value: 検証対象のパス
            info: フィールド情報：フィールド名などのメタデータ

        Returns:
            Path: 検証済みのパス

        Note:
            ファイルが存在しない場合でも起動は継続しますが、
            警告ログが出力されます。実際のエンドポイント実行時に
            適切なエラーハンドリングが必要です。
        """
        if not value.exists():
            logger.warning(
                f"設定されたパスが存在しません: {info.field_name}={value}. "
                f"該当機能の利用時にエラーが発生する可能性があります。"
            )
        elif not value.is_file():
            logger.warning(
                f"設定されたパスがファイルではありません: {info.field_name}={value}"
            )
        return value

    # `@model_validator` モデル全体の検証を行う、(mode="after") で個別フィールド検証後に実行
    @model_validator(mode="after")
    def _validate_startup_paths(self) -> "Settings":
        """起動時の統合的なパス検証を実行.

        Returns:
            Settings: 検証済みの設定インスタンス

        Note:
            個別のパス検証は@field_validatorで実施済みのため、
            ここでは追加の統合的な検証やログ出力を行います。
        """
        missing_paths = []
        for field_name in ["model_path", "scaler_path", "checkpoint_path"]:
            path = getattr(self, field_name)
            if not path.exists():
                missing_paths.append(f"{field_name}={path}")

        if missing_paths:
            logger.warning(
                f"一部のモデルファイルが見つかりません: {', '.join(missing_paths)}. "
                f"機械学習関連の機能が正常に動作しない可能性があります。"
            )
        else:
            logger.info("すべてのモデルファイルパスの検証が完了しました")

        return self
