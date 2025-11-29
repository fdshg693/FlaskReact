"""
YAML設定ファイルの解析に関するユーティリティ関数

PyYAMLを使用してYAMLファイルおよびフロントマターを解析します。
"""

from pathlib import Path
from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Markdownファイルのフロントマターを解析する

    フロントマターは `---` で囲まれたYAML形式のメタデータです。

    Args:
        content: Markdownファイルの内容

    Returns:
        tuple: (フロントマターの辞書, フロントマター以降の本文)

    Raises:
        ValueError: フロントマターが見つからない場合
    """
    lines = content.split("\n")

    # 最初の行が --- でなければフロントマターなし
    if not lines or lines[0].strip() != "---":
        raise ValueError("フロントマターが見つかりません")

    # 終了の --- を探す
    end_index = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break

    if end_index == -1:
        raise ValueError("フロントマターの終了が見つかりません")

    # フロントマター部分を抽出してパース
    frontmatter_text = "\n".join(lines[1:end_index])
    frontmatter = yaml.safe_load(frontmatter_text) or {}

    # 本文を抽出
    body = "\n".join(lines[end_index + 1 :])

    return frontmatter, body


def extract_custom_inputs(frontmatter: dict[str, Any]) -> dict[str, str]:
    """
    フロントマターからcustom_inputsを辞書形式で抽出する

    Args:
        frontmatter: フロントマターの辞書

    Returns:
        dict: {name: default} の辞書
    """
    custom_inputs: dict[str, str] = {}
    inputs_list = frontmatter.get("custom_inputs", [])

    for item in inputs_list:
        if isinstance(item, dict) and "name" in item:
            name = item["name"]
            default = item.get("default", "")
            custom_inputs[name] = default

    return custom_inputs


def extract_outputs(frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """
    フロントマターからoutputsセクションを抽出する

    Args:
        frontmatter: フロントマターの辞書

    Returns:
        list: outputsのリスト

    Raises:
        ValueError: outputsセクションが存在しない場合
    """
    outputs = frontmatter.get("outputs")
    if not outputs:
        raise ValueError("outputsセクションが存在しません")
    return outputs


def remove_custom_sections_from_frontmatter(
    frontmatter: dict[str, Any],
) -> dict[str, Any]:
    """
    フロントマターからcustom_inputsとoutputsセクションを削除する

    Args:
        frontmatter: フロントマターの辞書

    Returns:
        dict: custom_inputsとoutputsを除いたフロントマター
    """
    result = frontmatter.copy()
    result.pop("custom_inputs", None)
    result.pop("outputs", None)
    return result


class _FlowStyleListDumper(yaml.SafeDumper):
    """リストをフロースタイル（インライン形式）で出力するカスタムDumper"""

    pass


def _represent_list_as_flow(dumper: yaml.Dumper, data: list) -> yaml.Node:
    """リストをフロースタイルで表現する"""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_FlowStyleListDumper.add_representer(list, _represent_list_as_flow)


def rebuild_content_with_frontmatter(frontmatter: dict[str, Any], body: str) -> str:
    """
    フロントマターと本文を結合してMarkdownコンテンツを再構築する

    Args:
        frontmatter: フロントマターの辞書
        body: 本文

    Returns:
        str: 再構築されたMarkdownコンテンツ
    """
    if frontmatter:
        frontmatter_text = yaml.dump(
            frontmatter,
            Dumper=_FlowStyleListDumper,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        return f"---\n{frontmatter_text}\n---\n{body}"
    else:
        # フロントマターが空の場合は本文のみ
        return body.lstrip("\n")


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
