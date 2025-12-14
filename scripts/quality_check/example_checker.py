"""
プロジェクト配下の全ての.exampleファイルを取得
それらに対する、元のファイルが存在するかをチェックするスクリプト
存在しない場合は、ターミナルに警告を表示する

例：
- .pre-commit-config.yaml.example -> .pre-commit-config.yaml が存在するか

そして、.exampleファイルは全てGit管理下にあることを確認する
元のファイルは、GIT管理から外れていることを確認する

結果はscripts/output配下にMarkdown形式で出力される
"""

import subprocess
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """プロジェクトのルートディレクトリを取得する"""
    current = Path(__file__).resolve()
    # git repoのルートを探す
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    # 見つからない場合は現在のスクリプトの2階層上を返す
    return current.parent.parent.parent


def find_example_files(project_root: Path) -> list[Path]:
    """
    プロジェクト配下の全ての.exampleファイルを取得する
    .venvや.gitディレクトリは除外する
    """
    example_files = []
    exclude_dirs = {".venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache"}

    for path in project_root.rglob("*.example"):
        # 除外ディレクトリに含まれているかチェック
        if not any(excluded in path.parts for excluded in exclude_dirs):
            example_files.append(path)

    return sorted(example_files)


def get_original_file_path(example_file: Path) -> Path:
    """
    .exampleファイルから元のファイルパスを取得する
    例: config.yaml.example -> config.yaml
    """
    return example_file.with_suffix("")


def get_git_tracked_files(project_root: Path) -> set[Path]:
    """
    Git管理下にあるファイルの一覧を取得する
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = set()
        for line in result.stdout.strip().split("\n"):
            if line:
                tracked_files.add(project_root / line)
        return tracked_files
    except subprocess.CalledProcessError:
        print("警告: gitコマンドの実行に失敗しました")
        return set()


def check_example_files(
    project_root: Path,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """
    .exampleファイルのチェックを実行する

    Returns:
        tuple[list[str], list[str], list[dict[str, str]]]:
            (警告メッセージリスト, 情報メッセージリスト, ファイル情報リスト)
    """
    warnings: list[str] = []
    infos: list[str] = []
    file_infos: list[dict[str, str]] = []

    example_files = find_example_files(project_root)
    git_tracked_files = get_git_tracked_files(project_root)

    if not example_files:
        infos.append("✓ .exampleファイルは見つかりませんでした")
        return warnings, infos, file_infos

    infos.append(f"検出された.exampleファイル: {len(example_files)}件")
    infos.append("-" * 50)

    for example_file in example_files:
        relative_example = example_file.relative_to(project_root)
        original_file = get_original_file_path(example_file)
        relative_original = original_file.relative_to(project_root)

        # ファイル情報を収集
        file_info = {
            "example_path": str(relative_example),
            "original_path": str(relative_original),
            "original_exists": original_file.exists(),
            "example_in_git": example_file in git_tracked_files,
            "original_in_git": original_file in git_tracked_files,
        }
        file_infos.append(file_info)

        # 1. 元のファイルが存在するかチェック
        if not original_file.exists():
            warnings.append(
                f"⚠ 元のファイルが存在しません: {relative_original}\n"
                f"   (.exampleファイル: {relative_example})"
            )

        # 2. .exampleファイルがGit管理下にあるかチェック
        if example_file not in git_tracked_files:
            warnings.append(
                f"⚠ .exampleファイルがGit管理下にありません: {relative_example}"
            )

        # 3. 元のファイルがGit管理から外れているかチェック（存在する場合のみ）
        if original_file.exists() and original_file in git_tracked_files:
            warnings.append(
                f"⚠ 元のファイルがGit管理下にあります（除外推奨）: {relative_original}"
            )

    return warnings, infos, file_infos


def generate_markdown_report(
    project_root: Path,
    warnings: list[str],
    infos: list[str],
    file_infos: list[dict[str, str]],
) -> str:
    """
    Markdown形式のレポートを生成する
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Example File Check Report",
        "",
        f"**実行日時**: {timestamp}",
        f"**プロジェクトルート**: `{project_root}`",
        "",
        "---",
        "",
        "## 概要",
        "",
    ]

    for info in infos:
        lines.append(info)

    lines.extend(["", "---", "", "## ファイル一覧", ""])

    if file_infos:
        lines.append(
            "| # | Exampleファイル | 元のファイル | 元ファイル存在 | Example Git管理 | 元ファイル Git管理 |"
        )
        lines.append(
            "|---|----------------|-------------|:------------:|:--------------:|:-----------------:|"
        )

        for i, info in enumerate(file_infos, 1):
            original_exists = "✅" if info["original_exists"] else "❌"
            example_in_git = "✅" if info["example_in_git"] else "❌"
            original_in_git = "⚠️" if info["original_in_git"] else "✅"

            lines.append(
                f"| {i} | `{info['example_path']}` | `{info['original_path']}` | "
                f"{original_exists} | {example_in_git} | {original_in_git} |"
            )
    else:
        lines.append("_ファイルが見つかりませんでした_")

    lines.extend(["", "---", "", "## 警告", ""])

    if warnings:
        lines.append(f"**警告件数**: {len(warnings)}")
        lines.append("")
        for i, warning in enumerate(warnings, 1):
            # 複数行の警告を整形
            warning_lines = warning.split("\n")
            lines.append(f"### 警告 {i}")
            lines.append("")
            for wl in warning_lines:
                lines.append(f"> {wl}")
            lines.append("")
    else:
        lines.append("✅ **全てのチェックに合格しました**")

    lines.extend(["", "---", "", "## 凡例", ""])
    lines.append("- **元ファイル存在**: ✅ = 存在する, ❌ = 存在しない")
    lines.append("- **Example Git管理**: ✅ = Git管理下, ❌ = Git管理外（要対応）")
    lines.append(
        "- **元ファイル Git管理**: ✅ = Git管理外（正常）, ⚠️ = Git管理下（除外推奨）"
    )

    return "\n".join(lines)


def save_markdown_report(project_root: Path, content: str) -> Path:
    """
    Markdownレポートをファイルに保存する
    """
    output_dir = project_root / "scripts" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"example_check_{timestamp}.md"

    output_file.write_text(content, encoding="utf-8")
    return output_file


def main() -> int:
    """
    メイン関数

    Returns:
        int: 警告がある場合は1、ない場合は0
    """
    project_root = get_project_root()
    print(f"プロジェクトルート: {project_root}")
    print("=" * 50)

    warnings, infos, file_infos = check_example_files(project_root)

    # Markdownレポートを生成して保存
    markdown_content = generate_markdown_report(
        project_root, warnings, infos, file_infos
    )
    output_file = save_markdown_report(project_root, markdown_content)
    print(f"レポートを保存しました: {output_file}")

    # 情報メッセージを出力
    for info in infos:
        print(info)

    # 警告メッセージを出力
    if warnings:
        print("\n" + "=" * 50)
        print("【警告】")
        print("=" * 50)
        for warning in warnings:
            print(warning)
            print()
        print(f"警告件数: {len(warnings)}")
        return 1
    else:
        print("\n✓ 全てのチェックに合格しました")
        return 0


if __name__ == "__main__":
    exit(main())
