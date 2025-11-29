"""
YAML設定ファイルの解析に関するユーティリティ関数

標準ライブラリのみを使用したシンプルなYAMLパーサーを提供します。
"""

from pathlib import Path


def parse_yaml_include(yaml_path: Path) -> list[str]:
    """
    簡易YAMLパーサー：includeセクションのリストを取得する

    標準ライブラリのみを使用するため、シンプルな形式のみ対応。
    コメント行と空行は無視されます。

    対応形式:
        include:
          - pattern1
          - pattern2  # コメントも除去される

    Args:
        yaml_path: 解析対象のYAMLファイルパス

    Returns:
        list[str]: includeセクションのパターンリスト
    """
    patterns: list[str] = []
    in_include_section = False

    with yaml_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # コメントまたは空行はスキップ
            if not stripped or stripped.startswith("#"):
                continue

            # includeセクションの開始を検出
            if stripped.startswith("include:"):
                in_include_section = True
                continue

            # 別のセクションが始まったら終了
            if (
                not line.startswith(" ")
                and not line.startswith("\t")
                and ":" in stripped
            ):
                in_include_section = False
                continue

            # includeセクション内のリストアイテムを解析
            if in_include_section and stripped.startswith("-"):
                # コメント部分を除去
                item = stripped[1:].split("#")[0].strip()
                # クォートを除去
                item = item.strip("'\"")
                if item:
                    patterns.append(item)

    return patterns
