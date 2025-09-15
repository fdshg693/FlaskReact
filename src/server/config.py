from __future__ import annotations

from pathlib import Path
from typing import List, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FLASKREACT_",
        extra="ignore",
    )
    # CORS
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # Upload limits and allowed types
    allowed_image_extensions: Set[str] = Field(
        default_factory=lambda: {"png", "jpg", "jpeg", "gif"}
    )
    allowed_pdf_extensions: Set[str] = Field(default_factory=lambda: {"pdf"})
    max_image_size_mb: int = 5
    max_pdf_size_mb: int = 10

    # Model paths
    app_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    model_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "param"
        / "models_20250712_021710.pth"
    )
    scaler_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "scaler"
        / "scaler.joblib"
    )
    checkpoint_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "checkpoints"
        / "2025_09_06_20_49_09_img128_layer3_hidden4096_3class_dropout0.2_scale1.5_test_dataset"
        / "best_accuracy.pth"
    )

    # No inner Config (deprecated). Use model_config above.
