"""
プロジェクト配下の全ての.exampleファイルを取得
それらに対する、元のファイルが存在するかをチェックするスクリプト
存在しない場合は、ターミナルに警告を表示する

例：
- .pre-commit-config.yaml.example -> .pre-commit-config.yaml が存在するか

そして、.exampleファイルは全てGit管理下にあることを確認する
元のファイルは、GIT管理から外れていることを確認する

"""

from pathlib import Path
import subprocess


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


def check_example_files(project_root: Path) -> tuple[list[str], list[str]]:
    """
    .exampleファイルのチェックを実行する

    Returns:
        tuple[list[str], list[str]]: (警告メッセージリスト, 情報メッセージリスト)
    """
    warnings: list[str] = []
    infos: list[str] = []

    example_files = find_example_files(project_root)
    git_tracked_files = get_git_tracked_files(project_root)

    if not example_files:
        infos.append("✓ .exampleファイルは見つかりませんでした")
        return warnings, infos

    infos.append(f"検出された.exampleファイル: {len(example_files)}件")
    infos.append("-" * 50)

    for example_file in example_files:
        relative_example = example_file.relative_to(project_root)
        original_file = get_original_file_path(example_file)
        relative_original = original_file.relative_to(project_root)

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

    return warnings, infos


def main() -> int:
    """
    メイン関数

    Returns:
        int: 警告がある場合は1、ない場合は0
    """
    project_root = get_project_root()
    print(f"プロジェクトルート: {project_root}")
    print("=" * 50)

    warnings, infos = check_example_files(project_root)

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
