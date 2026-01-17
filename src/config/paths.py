"""
パス設定モジュール。
設定ファイル: src/config/pahts.ini
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

# シングルトンビルド中フラグ
# 基本FALSEに設定して、このファイル内のシングルトンインスタンス生成の直前にTRUEに設定する（生成後はFALSEに戻す）
# シングルトンPROJECTPATHSが唯一のインスタンスであることを保証し、直接インスタンス化を警告するために使用する
_BUILDING_SINGLETON = False


class _PATHS(BaseModel):
    """
    プロジェクトパスを格納したシングルトンPROJECTPATHSを作成するためのクラス。

    Note;
        直接のインスタンス化は行わない。
        不変オブジェクトとして振る舞う。

    Examples:
        >>> from config.paths import PROJECTPATHS
        >>> print(PROJECTPATHS.data)
        /path/to/project/data
    """

    # このプロジェクトのルートパス
    project_root: Path

    # ルート直下の主要ディレクトリ
    src: Path
    data: Path
    frontend: Path
    flask_static: Path
    logs: Path

    # データ関連パス
    original_data: Path
    outputs: Path
    tmp: Path
    ml_data: Path
    iris_data_path: Path
    diabetes_data_path: Path
    titanic_test_data_path: Path
    titanic_train_data_path: Path

    # ML出力関連パス
    ml_image_data: Path
    ml_learning_curves_dir: Path
    ml_logs: Path

    # サーバー側で利用するデフォルトモデルパス
    default_iris_model_path: Path
    default_iris_scaler_path: Path

    llm_data: Path

    # Pydanticモデルとしての振る舞いの設定
    # arbitrary_types_allowed=True -> Path型などのPydanticモデル外の型を許可
    # frozen=True -> インスタンスを不変にする
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # model_validatorは、モデル全体のバリデーションを行うためのデコレータ。
    # mode="after"は、全てのフィールドのバリデーションが終わった後にこのメソッドが呼び出されることを意味する。
    @model_validator(mode="after")
    def _validate_paths(self) -> _PATHS:
        """各パスが存在するかをチェック。存在しない場合は警告を出力。"""
        global _BUILDING_SINGLETON  # noqa: PLW0603

        # シングルトン以外での直接インスタンス化を警告
        if not _BUILDING_SINGLETON:
            import warnings

            warnings.warn(
                "Direct instantiation of Paths is discouraged. "
                "Use the module-level PATHS singleton instead: from config import PATHS",
                UserWarning,
                stacklevel=3,
            )

        # パス存在チェック - 存在しない場合は警告をコンソール出力
        for field_name, field_value in self:
            if isinstance(field_value, Path):
                if not field_value.exists():
                    logger.warning(
                        "Path does not exist: {} = {}", field_name, field_value
                    )

        return self

    @classmethod
    def _build(cls, root: Path | None = None, config_path: str | None = None) -> _PATHS:
        """
        シングルトンPROJECTPATHSを構築するためのクラスメソッド。
        外部からこのメソッドを呼び出してインスタンス化しないこと。

        Args:
            root: このプロジェクトのルートパス。Noneの場合、自動検出される。
                  Noneの場合、環境変数PROJECT_ROOTから取得し、設定されていない場合はこのファイルの2階層上をルートとする。
            config_path: 設定ファイルの相対パス（プロジェクトルートからの相対パス）。
                        デフォルトは "src/config/pahts.ini"。

        Returns:
            Paths: 構築されたPathsインスタンス。

        Raises:
            RuntimeError: プロジェクトルートが特定できない場合、または設定ファイルが見つからない場合に発生。
        """
        if root is None:
            env_root = os.getenv("PROJECT_ROOT")
            if env_root:
                root = Path(env_root).resolve()
                logger.debug("Using PROJECT_ROOT from env: {}", root)
            else:
                # repo root/src/config/paths.py -> parents[2] == repo root/
                root = Path(__file__).resolve().parents[2]
                logger.debug("Using inferred project root: {}", root)

        if not root.exists():
            msg = f"Project root does not exist: {root}"
            logger.error(msg)
            raise RuntimeError(msg)

        # 設定ファイルパスの決定
        if config_path is None:
            config_path = "src/config/pahts.ini"

        config_file = root / config_path

        if not config_file.exists():
            msg = f"Config file does not exist: {config_file}"
            logger.error(msg)
            raise RuntimeError(msg)

        # 設定ファイルを読み込む
        logger.info("Loading configuration from: {}", config_file)
        config = configparser.ConfigParser()
        config.read(config_file)

        # paths セクションから読み込み
        if "paths" not in config:
            msg = f"Config file missing [paths] section: {config_file}"
            logger.error(msg)
            raise RuntimeError(msg)

        paths_section = config["paths"]

        # 相対パスを絶対パスに変換
        def resolve_path(relative_path: str) -> Path:
            """相対パスをプロジェクトルートからの絶対パスに変換"""
            # ダブルクォートを削除
            relative_path = relative_path.strip("\"'")
            return (root / relative_path).resolve()

        # 設定ファイルから各パスを読み込み
        # クラスフィールドごとにループして path_dict を構築
        try:
            path_dict = {}
            for field_name in cls.model_fields.keys():
                if field_name == "project_root":
                    # project_root は設定ファイルにないため、直接指定
                    path_dict[field_name] = root
                else:
                    # 設定ファイルから同一名の値を取得
                    relative_path = paths_section.get(field_name)
                    if relative_path is None:
                        msg = f"Missing required path in config: {field_name}"
                        logger.error(msg)
                        raise RuntimeError(msg)
                    path_dict[field_name] = resolve_path(relative_path)
        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Error parsing config file: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        logger.info("Building Paths instance from root: {}", root)

        return cls(**path_dict)


# このファイルのこの部分でのみシングルトンインスタンスを構築可能にする
_BUILDING_SINGLETON = True
PROJECTPATHS: _PATHS = _PATHS._build()
_BUILDING_SINGLETON = False


def get_path(
    *parts: str | Path, root: Path | None = None, create: bool = False
) -> Path:
    """
    ルートからの相対パスを結合してPathオブジェクトを取得するユーティリティ関数。

    Args:
        *parts: 結合するパスの部分となり、カンマ区切りで任意の数だけ指定可能。拡張子付きファイル名は含めないでください。
        root: ベースとなるディレクトリパス（デフォルトはPATHS.project_root）。
        create: ディレクトリ構造を作成する場合はTrue。

    Returns:
        解決されたPathオブジェクト。

    Raises:
        OSError: ディレクトリ作成に失敗した場合。

    Note:
        ファイルパスを含める場合は、ensure_path_exists関数を使用して親ディレクトリを作成してください。

    Example:
        >>> # デフォルトのプロジェクトルートからdata/new_dirを取得して、ディレクトリを作成
        >>> p = get_path("data", "new_dir", create=True)

        >>> # カスタムルートからoutputs/resultsを取得
        >>> p = get_path(Path("outputs"), "results", root=PATHS.custom_root)
    """
    base: Path = root or PROJECTPATHS.project_root
    joined_path: Path = base.joinpath(*[str(part) for part in parts])

    if create:
        try:
            joined_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Created directory: {}", joined_path)
        except OSError as e:
            logger.error("Failed to create directory {}: {}", joined_path, e)
            raise

    return joined_path


def find_paths(
    pattern: str, root: Path | None = None, recursive: bool = True
) -> list[Path]:
    """
    ルート配下にあるパスをグロブパターンで検索するユーティリティ関数。

    Args:
        pattern: グロブパターン（例: "*.py", "**/*.json"）。
        root: ベースとなるパス（デフォルトはPATHS.project_root）。
        recursive: 再帰的に検索する場合はTrue。(rglobを使用)

    Returns:
        一致するPathオブジェクトのリスト。

    Example:
        >>> py_files = find_paths("*.py", recursive=False)
        >>> all_json = find_paths("**/*.json")
    """
    base: Path = root or PROJECTPATHS.project_root

    try:
        # rglobは再帰的に全てのサブディレクトリを探索
        # globはデフォルトで現在のディレクトリのみ探索、**を使うと再帰的に探索可能
        if recursive:
            results: list[Path] = list(base.rglob(pattern))
        else:
            results: list[Path] = list(base.glob(pattern))
        logger.debug("Found {} paths matching '{}' in {}", len(results), pattern, base)
        return results
    except Exception as e:
        logger.error(
            "Error finding paths with pattern '{}' in {}: {}", pattern, base, e
        )
        return []


def ensure_path_exists(path: Path, *, is_file: bool = False) -> Path:
    """
    指定されたパス（ディレクトリまたはファイルの親ディレクトリ）が存在することを保証するユーティリティ関数。

    Args:
        path: ファイルパスの場合は、親ディレクトリの存在を、ディレクトリパスの場合はそのディレクトリ自体の存在を保証します。ただし、既に存在する場合は何もしません。
        is_file: ファイルパスの場合はTrueに設定してください。
                 Trueの場合、親ディレクトリのみ作成し、ファイル自体は作成しません。

    Returns:
        指定されたPathオブジェクト。（チェーン可能とするために返されます）

    Example:
        >>> log_file = ensure_path_exists(PATHS.logs / "app.log", is_file=True)
            これは、logsディレクトリが存在しない場合に作成しますが、app.logファイル自体は作成しません。
        >>> data_dir = ensure_path_exists(PATHS.data / "cache")
            これは、data/cacheディレクトリが存在しない場合に作成します。
    """
    if is_file:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured parent directory exists for file: {}", path)
    else:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: {}", path)

    return path


__all__ = ["PROJECTPATHS", "get_path", "find_paths", "ensure_path_exists"]
