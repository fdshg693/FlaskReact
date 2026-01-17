"""PROJECTPATHS から `paths.txt` を生成するユーティリティ。

`config.paths.PROJECTPATHS` に定義されたパス（主に `Path` 型）を走査し、
プロジェクトルートからの相対ツリーとして `src/config/paths.txt` に書き出す。

ツリーの各ファイル/ディレクトリ名の右側に、そのパスに対応するプロパティ名を
`(...)` で併記する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import PROJECTPATHS


@dataclass(slots=True)
class _Node:
    """ツリー描画用のノード。

    - `dirs`: 子ディレクトリ
    - `files`: 直下のファイル名
    - `labels`: このディレクトリ自体を指す PROJECTPATHS のフィールド名
    - `file_labels`: 直下ファイル名 -> そのファイルを指すフィールド名の集合
    """

    dirs: dict[str, "_Node"] = field(default_factory=dict)
    files: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    file_labels: dict[str, set[str]] = field(default_factory=dict)


def _is_file_path(field_name: str, path: Path) -> bool:
    """`PROJECTPATHS` のフィールドがファイルを指すかどうかを判定する。"""

    return field_name.endswith("_path") or (path.suffix != "")


def _rel_parts(path: Path, root: Path) -> tuple[str, ...]:
    """`path` を `root` からの相対パスに分解し、各パーツのタプルで返す。"""

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(os.path.relpath(path.as_posix(), root.as_posix()))
    return tuple(part for part in rel.parts if part not in (".", ""))


def _add_path(
    root_node: _Node,
    parts: tuple[str, ...],
    *,
    is_file: bool,
    field_name: str,
) -> None:
    """相対パス `parts` をツリーへ追加し、該当するフィールド名を紐付ける。"""

    if not parts:
        return

    node = root_node
    if is_file:
        for part in parts[:-1]:
            node = node.dirs.setdefault(part, _Node())
        node.files.add(parts[-1])
        node.file_labels.setdefault(parts[-1], set()).add(field_name)
        return

    for part in parts:
        node = node.dirs.setdefault(part, _Node())

    node.labels.add(field_name)


def _render_tree(root_node: _Node) -> list[str]:
    """ツリーを `tree` 風の文字列リストとして描画する。"""

    lines: list[str] = ["."]

    def format_labels(labels: set[str]) -> str:
        if not labels:
            return ""
        return f" ({', '.join(sorted(labels))})"

    def walk(node: _Node, prefix: str) -> None:
        dir_names = sorted(node.dirs.keys())
        file_names = sorted(node.files)
        entries: list[tuple[str, bool]] = [(d, True) for d in dir_names] + [
            (f, False) for f in file_names
        ]

        for i, (name, is_dir) in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            if is_dir:
                child = node.dirs[name]
                display_name = f"{name}/" + format_labels(child.labels)
            else:
                display_name = name + format_labels(node.file_labels.get(name, set()))

            lines.append(f"{prefix}{connector}{display_name}")

            if is_dir:
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(node.dirs[name], child_prefix)

    walk(root_node, "")
    return lines


def main() -> None:
    """PROJECTPATHS を走査して `src/config/paths.txt` を生成する。"""

    project_root: Path = PROJECTPATHS.project_root
    root_node = _Node()

    for field_name in sorted(type(PROJECTPATHS).model_fields.keys()):
        if field_name == "project_root":
            continue
        value = getattr(PROJECTPATHS, field_name, None)
        if not isinstance(value, Path):
            continue
        parts = _rel_parts(value, project_root)
        if not parts:
            continue
        _add_path(
            root_node,
            parts,
            is_file=_is_file_path(field_name, value),
            field_name=field_name,
        )

    out_path = Path(__file__).with_name("paths.txt")
    lines = [
        f"# Auto-generated from PROJECTPATHS by {Path(__file__).name}",
        f"# Root: {project_root}",
        *_render_tree(root_node),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
