from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# シングルトンビルド中フラグ
# 基本FALSEに設定して、このファイル内のシングルトンインスタンス生成の直前にTRUEに設定する（生成後はFALSEに戻す）
# シングルトンPROJECTPATHSが唯一のインスタンスであることを保証し、直接インスタンス化を警告するために使用する
_BUILDING_SINGLETON = False


class Paths(BaseModel):
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

    # root
    project_root: Path

    # top-level dirs
    src: Path
    data: Path
    frontend: Path
    flask_static: Path
    logs: Path

    # dataset dirs/files
    original_data: Path
    outputs: Path
    tmp: Path
    ml_data: Path
    iris_data_path: Path
    diabetes_data_path: Path
    titanic_test_data_path: Path
    titanic_train_data_path: Path

    # ML outputs
    ml_outputs: Path
    ml_image_data: Path
    ml_learning_curves_dir: Path

    llm_data: Path

    # Pydanticモデルとしての振る舞いの設定
    # arbitrary_types_allowed=True -> Path型などのPydanticモデル外の型を許可
    # frozen=True -> インスタンスを不変にする
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # field_validatorは、モデルのフィールドごとにバリデーションを行うためのデコレータ。
    # "*"は全フィールドを意味するワイルドカード。全てのPathフィールドに対して適用される。
    # mode="before"は、自動型変換などのバリデーションが行われる前にこのメソッドが呼び出されることを意味する。
    @field_validator("*", mode="before")
    @classmethod
    def _ensure_path_type(cls, value: object) -> Path:
        """
        全てのPathフィールドに対して、strが渡された場合にPathに変換するバリデータ。
        """
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            import warnings

            warnings.warn(
                f"Converting string to Path (prefer passing Path objects): {value}",
                UserWarning,
                stacklevel=3,
            )
            return Path(value)
        msg = f"Expected Path or str, got {type(value).__name__}"
        raise TypeError(msg)

    # model_validatorは、モデル全体のバリデーションを行うためのデコレータ。
    # mode="after"は、全てのフィールドのバリデーションが終わった後にこのメソッドが呼び出されることを意味する。
    @model_validator(mode="after")
    def _validate_paths(self) -> Paths:
        """Validate that all paths are properly formed."""
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

        # Additional validation: ensure project_root is absolute
        if not self.project_root.is_absolute():
            logger.warning("project_root is not absolute: {}", self.project_root)
            # Convert to absolute
            object.__setattr__(self, "project_root", self.project_root.resolve())

        logger.debug("Paths instance validated successfully")
        return self

    @classmethod
    def _build(cls, root: Path | None = None) -> Paths:
        """
        シングルトンPROJECTPATHSを構築するためのクラスメソッド。
        外部からこのメソッドを呼び出してインスタンス化しないこと。

        Args:
            root: このプロジェクトのルートパス。Noneの場合、自動検出される。
            Noneの場合、環境変数PROJECT_ROOTから取得し、設定されていない場合はこのファイルの2階層上をルートとする。

        Returns:
            Paths: 構築されたPathsインスタンス。

        Raises:
            RuntimeError: プロジェクトルートが特定できない場合に発生。
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

        # Build all paths
        src = root / "src"

        # フロントエンド
        frontend = root / "frontend"

        # Flaskサーバー利用静的ファイル
        flask_static = frontend / "flask_static"

        # ログファイル格納ディレクトリ
        logs = root / "logs"

        # data関連
        data = root / "data"
        # =============
        # 一次データ
        # =============
        original_data = data / "original_sources"

        ml_data = original_data / "ml"

        iris_data_path = ml_data / "iris" / "iris.csv"
        diabetes_data_path = ml_data / "diabetes" / "diabetes.csv"
        titanic_test_data_path = ml_data / "others" / "titanic_test.csv"
        titanic_train_data_path = ml_data / "others" / "titanic_train.csv"

        llm_data = original_data / "llm"

        # =============
        # 生成データ（参照・加工用）
        # =============
        outputs = data / "outputs"

        ml_outputs = outputs / "ml"

        # テスト用画像の出力先
        ml_image_data = ml_outputs / "image"

        ml_learning_curves_dir = ml_outputs / "learning_curves"

        # =============
        # 一時出力データ（参照・加工されることを想定しない）
        # =============
        tmp = data / "tmp"

        logger.info("Building Paths instance from root: {}", root)

        return cls(
            project_root=root,
            src=src,
            data=data,
            frontend=frontend,
            original_data=original_data,
            outputs=outputs,
            tmp=tmp,
            flask_static=flask_static,
            logs=logs,
            ml_data=ml_data,
            iris_data_path=iris_data_path,
            diabetes_data_path=diabetes_data_path,
            titanic_test_data_path=titanic_test_data_path,
            titanic_train_data_path=titanic_train_data_path,
            ml_outputs=ml_outputs,
            ml_image_data=ml_image_data,
            ml_learning_curves_dir=ml_learning_curves_dir,
            llm_data=llm_data,
        )


# このファイルのこの部分でのみシングルトンインスタンスを構築可能にする
_BUILDING_SINGLETON = True
PROJECTPATHS: Paths = Paths._build()
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
        recursive: 再帰的に検索する場合はTrue。

    Returns:
        一致するPathオブジェクトのリスト。

    Example:
        >>> py_files = find_paths("*.py", recursive=False)
        >>> all_json = find_paths("**/*.json")
    """
    base: Path = root or PROJECTPATHS.project_root

    try:
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
        path: 存在を保証するパス。（ファイルパス・ディレクトリパスのいずれか）
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


__all__ = ["Paths", "PROJECTPATHS", "get_path", "find_paths", "ensure_path_exists"]
