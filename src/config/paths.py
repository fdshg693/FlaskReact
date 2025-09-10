from pathlib import Path
from typing import Union


def get_path(*parts: Union[str, Path], root: Path, create: bool = False) -> Path:
    """root に対して遅延的にパスを結合して返す。必要なら作成する。"""
    p = root.joinpath(*[str(p) for p in parts])
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def find_paths(pattern: str, root: Path, recursive: bool = True) -> list[Path]:
    """glob ベースで検索。呼び出し側で必要に応じて絞る。"""
    if recursive:
        return list(root.rglob(pattern))
    return list(root.glob(pattern))
