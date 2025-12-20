"""
パス操作に関するユーティリティ関数

プロジェクトルートの取得や、パス変換などの共通処理を提供します。
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    プロジェクトのルートディレクトリを取得する

    カレントワーキングディレクトリ（CWD）を基準とし、
    pyproject.tomlが存在するディレクトリをプロジェクトルートとして判定します。

    Returns:
        Path: プロジェクトルートのパス
    """
    cwd = Path.cwd().resolve()

    # CWDから親を辿ってpyproject.tomlを探す
    current = cwd
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # pyproject.tomlが見つからない場合はCWDを返す
    return cwd


def path_to_dot_notation(path: Path, base_dir: Path) -> str:
    """
    パスをドット区切りの文字列に変換する

    Args:
        path: 変換対象のパス
        base_dir: 基準となるディレクトリ

    Returns:
        str: ドット区切りの文字列（例: general.basic）
    """
    relative_path = path.relative_to(base_dir)
    return str(relative_path).replace("/", ".").replace("\\", ".")
