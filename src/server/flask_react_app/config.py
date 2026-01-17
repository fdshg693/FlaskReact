from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Set

from loguru import logger
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config import PROJECTPATHS, load_dotenv_workspace


def init_env() -> None:
    """Load .env once without overriding existing system env vars."""

    load_dotenv_workspace(override=False)


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return a process-wide Settings singleton (validated once)."""

    init_env()
    return Settings()


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

    # NOTE: ".env" is loaded once at startup via config.load_setting.load_dotenv_workspace().
    # Keep system env precedence by using override=False there (do not use Pydantic env_file).
    model_config = SettingsConfigDict(
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
    app_root: Path = Field(default_factory=lambda: PROJECTPATHS.src)
