from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


def read_csv_into_dataframe(
    path: str | Path, *, encoding: Optional[str] = None
) -> pd.DataFrame:
    """ファイルパスから CSV を読み込み、pandas DataFrame を返します。

    パス処理は pathlib.Path を使い、ログは loguru を使ってプロジェクトの
    既存慣習に合わせています。

    Args:
        path: CSV ファイルのパス（Path または str）。
        encoding: 文字エンコーディング（pandas.read_csv に渡されます）。

    Returns:
        読み込んだ pandas.DataFrame。

    Raises:
        FileNotFoundError: パスが存在しない、またはファイルではない場合。
        pd.errors.EmptyDataError: CSV が空の場合。
        Exception: pandas のパースエラー等はそのまま送出します。
    """
    p = Path(path)
    logger.debug("read_csv_from_path() -> resolving path: {}", p)

    if not p.exists() or not p.is_file():
        logger.error("CSV file not found at path: {}", p)
        raise FileNotFoundError(f"CSV file not found: {p}")

    try:
        df = pd.read_csv(p, encoding=encoding) if encoding else pd.read_csv(p)  # pyright: ignore[reportUnknownMemberType]
        logger.info("Loaded CSV with shape {} from {}", df.shape, p)
        return df
    except Exception:  # 呼び出し側で pandas 例外等を扱えるよう広めに捕捉
        logger.exception("Failed to read CSV from %s", p)
        raise


def save_csv_to_path(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    overwrite: bool = True,
    header: bool | list[str] | None = True,
    encoding: Optional[str] = None,
) -> Path:
    """pandas DataFrame を CSV として保存します（親ディレクトリは作成）。

    Args:
        df: 保存する DataFrame。
        path: 保存先ファイルパス（str または Path）。
        index: 行インデックスを書き込むかどうか。
        overwrite: False かつファイルが存在する場合は FileExistsError。
        header: 列名を書き込むか、またはカスタム列名リスト。
        encoding: 文字エンコーディング（DataFrame.to_csv に渡されます）。

    Returns:
        保存した CSV ファイルの絶対パス（Path）。

    Raises:
        FileExistsError: ファイルが存在し、overwrite が False の場合。
        Exception: IO や pandas のエラーはそのまま送出します。
    """
    p = Path(path)
    logger.debug("save_csv_to_path() -> target path: {}", p)

    if p.exists() and not overwrite:
        logger.error("File already exists and overwrite=False: {}", p)
        raise FileExistsError(f"File exists and overwrite is False: {p}")

    # 親ディレクトリがなければ作成
    if not p.parent.exists():
        logger.debug("Creating parent directories for {}", p.parent)
        p.parent.mkdir(parents=True, exist_ok=True)

    try:
        # header/encoding の指定を尊重して保存
        to_csv_kwargs = {"index": index, "header": header}
        if encoding:
            to_csv_kwargs["encoding"] = encoding
        df.to_csv(p, **to_csv_kwargs)
        logger.info("Saved CSV with shape {} to {}", getattr(df, "shape", "?"), p)
        return p.resolve()
    except Exception:
        logger.exception("Failed to save CSV to %s", p)
        raise
