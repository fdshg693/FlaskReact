# path.env の内容を元にキーに対応するパスを返すユーティリティ
# 仕様:
#  - プロジェクトルート直下の path.env を読み込む (KEY=relative/or/absolute/path)
#  - コメント(#開始)と空行は無視
#  - 一度読み込んだらキャッシュする
#  - 取得時に ~, 環境変数, 相対パス を解決し Path を返す（絶対パス）
#  - キーが存在しなければ KeyError, path.env が無ければ FileNotFoundError
#  - パスが存在しない場合は警告ログを出しそのまま返す（作成はしない）

from __future__ import annotations

from pathlib import Path
from typing import Dict

from loguru import logger

_PATH_CACHE: Dict[str, Path] = {}
_LOADED = False


def _project_root() -> Path:
    # src/util/resolve_path.py -> parents[2] = プロジェクトルート想定
    return Path(__file__).resolve().parents[2]


def _path_env_file() -> Path:
    # 環境変数で上書き可能 (例: export PATH_ENV_FILE=/custom/path.env)
    env_val = None
    try:
        import os

        env_val = os.environ.get("PATH_ENV_FILE")
    except Exception:  # pragma: no cover - 環境依存のため
        env_val = None
    if env_val:
        return Path(env_val).expanduser()
    return _project_root() / "path.env"


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    path_file = _path_env_file()
    if not path_file.exists():
        raise FileNotFoundError(f"path.env file not found: {path_file}")

    logger.debug(f"Loading path definitions from {path_file}")
    for line in path_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            logger.warning(f"Invalid line in path.env (ignored): {line}")
            continue
        k, v = stripped.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        resolved = _resolve_single_path(v)
        _PATH_CACHE[k] = resolved
    _LOADED = True
    logger.debug(f"Loaded {len(_PATH_CACHE)} path keys: {list(_PATH_CACHE.keys())}")


def _resolve_single_path(raw: str) -> Path:
    import os

    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded)
    if not p.is_absolute():
        p = _project_root() / p
    return p.resolve()


def resolve_path(key: str) -> Path:
    """指定キーに対応するパスを返す。

    Args:
            key: path.env に定義されたキー名 (例: "TEXT_DOCUMENTS_PATH")

    Returns:
            解決済み (絶対) Path オブジェクト

    Raises:
            FileNotFoundError: path.env が存在しない場合
            KeyError: キーが存在しない場合
    """

    if not key:
        raise KeyError("Empty key is not allowed")
    _load()
    if key not in _PATH_CACHE:
        raise KeyError(f"Key '{key}' not found in path.env")
    p = _PATH_CACHE[key]
    if not p.exists():
        logger.warning(f"Path for key '{key}' does not exist: {p}")
    return p


__all__ = ["resolve_path"]
