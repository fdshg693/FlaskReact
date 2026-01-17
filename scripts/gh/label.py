#!/usr/bin/env python3
"""
GitHub ラベル同期スクリプト

docs/problems/label.yaml に定義されたラベルと、
GitHubリポジトリの現在のラベルを比較・同期する。

Usage:
    python label.py [check|sync]

    check: 差分を表示するのみ（デフォルト）
    sync:  差分を解消してlabel.yamlに一致させる
"""

import json
import subprocess
import sys
from pathlib import Path


def load_yaml_labels(yaml_path: Path) -> dict[str, str]:
    """YAMLファイルからラベル定義を読み込む（標準ライブラリのみ使用）"""
    labels = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, description = line.split(":", 1)
                labels[name.strip()] = description.strip()
    return labels


def get_github_labels() -> dict[str, str]:
    """ghコマンドでリポジトリの現在のラベルを取得"""
    result = subprocess.run(
        ["gh", "label", "list", "--json", "name,description"],
        capture_output=True,
        text=True,
        check=True,
    )
    label_list = json.loads(result.stdout)
    return {item["name"]: item.get("description", "") for item in label_list}


def compare_labels(
    yaml_labels: dict[str, str], github_labels: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], dict[str, tuple[str, str]]]:
    """ラベルの差分を比較

    Returns:
        to_create: 作成が必要なラベル {name: description}
        to_delete: 削除が必要なラベル {name: description}
        to_update: 更新が必要なラベル {name: (old_desc, new_desc)}
    """
    yaml_names = set(yaml_labels.keys())
    github_names = set(github_labels.keys())

    # 作成が必要（YAMLにあってGitHubにない）
    to_create = {name: yaml_labels[name] for name in yaml_names - github_names}

    # 削除が必要（GitHubにあってYAMLにない）
    to_delete = {name: github_labels[name] for name in github_names - yaml_names}

    # 更新が必要（両方にあるが説明が異なる）
    to_update = {}
    for name in yaml_names & github_names:
        if yaml_labels[name] != github_labels[name]:
            to_update[name] = (github_labels[name], yaml_labels[name])

    return to_create, to_delete, to_update


def print_diff(
    to_create: dict[str, str],
    to_delete: dict[str, str],
    to_update: dict[str, tuple[str, str]],
) -> bool:
    """差分を表示。差分があればTrueを返す"""
    has_diff = bool(to_create or to_delete or to_update)

    if not has_diff:
        print("✅ ラベルは同期されています")
        return False

    print("📋 ラベル差分:")
    print()

    if to_create:
        print("➕ 作成が必要:")
        for name, desc in to_create.items():
            print(f"   {name}: {desc}")
        print()

    if to_delete:
        print("➖ 削除が必要:")
        for name, desc in to_delete.items():
            print(f"   {name}: {desc}")
        print()

    if to_update:
        print("📝 説明の更新が必要:")
        for name, (old, new) in to_update.items():
            print(f"   {name}:")
            print(f"      現在: {old}")
            print(f"      変更後: {new}")
        print()

    return True


def sync_labels(
    to_create: dict[str, str],
    to_delete: dict[str, str],
    to_update: dict[str, tuple[str, str]],
) -> None:
    """ghコマンドでラベルを同期"""
    # 新規作成
    for name, desc in to_create.items():
        print(f"➕ 作成中: {name}")
        subprocess.run(
            ["gh", "label", "create", name, "--description", desc],
            check=True,
        )

    # 削除
    for name in to_delete:
        print(f"➖ 削除中: {name}")
        subprocess.run(
            ["gh", "label", "delete", name, "--yes"],
            check=True,
        )

    # 更新
    for name, (_, new_desc) in to_update.items():
        print(f"📝 更新中: {name}")
        subprocess.run(
            ["gh", "label", "edit", name, "--description", new_desc],
            check=True,
        )


def main():
    # 引数の解析
    mode = "check"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ("check", "sync"):
            print(f"エラー: 不明なモード '{mode}'")
            print("Usage: python label.py [check|sync]")
            sys.exit(1)

    # label.yamlのパスを解決
    script_dir = Path(__file__).parent
    yaml_path = script_dir.parent.parent / "docs" / "problems" / "label.yaml"

    if not yaml_path.exists():
        print(f"エラー: {yaml_path} が見つかりません")
        sys.exit(1)

    # ラベル情報を取得
    print("📂 label.yaml を読み込み中...")
    yaml_labels = load_yaml_labels(yaml_path)
    print(f"   {len(yaml_labels)} 件のラベル定義")

    print("🌐 GitHub ラベルを取得中...")
    github_labels = get_github_labels()
    print(f"   {len(github_labels)} 件のラベル")
    print()

    # 差分を比較
    to_create, to_delete, to_update = compare_labels(yaml_labels, github_labels)

    # 差分を表示
    has_diff = print_diff(to_create, to_delete, to_update)

    # syncモードの場合は同期を実行
    if mode == "sync" and has_diff:
        print("🔄 同期を開始します...")
        print()
        sync_labels(to_create, to_delete, to_update)
        print()
        print("✅ 同期が完了しました")


if __name__ == "__main__":
    main()
