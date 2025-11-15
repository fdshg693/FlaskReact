from __future__ import annotations

from pathlib import Path
from typing import List, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

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
        / "models_20250712_021710.pth"
    )
    scaler_path: Path = Field(
        default_factory=lambda: PATHS.ml_outputs / "scaler" / "scaler.joblib"
    )
    checkpoint_path: Path = Field(
        default_factory=lambda: PATHS.ml_outputs
        / "checkpoints"
        / "2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset"
        / "best_accuracy.pth"
    )
