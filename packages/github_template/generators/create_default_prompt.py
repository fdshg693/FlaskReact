"""
.github_copilot_template配下の最下層ディレクトリにdefault.prompt.mdを自動生成するスクリプト

==============================================================================
概要
==============================================================================
このスクリプトは、.github_copilot_template/配下の全ての「最下層ディレクトリ」
（子ディレクトリを持たないディレクトリ）を探索し、各ディレクトリに
default.prompt.mdファイルを作成します。

==============================================================================
生成されるファイルの例
==============================================================================
ディレクトリ構造:
    .github_copilot_template/general/basic/
    .github_copilot_template/coder/script/

生成されるファイル:
    .github_copilot_template/general/basic/default.prompt.md
    .github_copilot_template/coder/script/default.prompt.md

ファイルの内容（general/basicの場合）:
    ---
    agent: general.basic
    ---
    read .github/tasks/general.basic.md to understand your task.

==============================================================================
使用方法
==============================================================================
    python scripts/github_copilot/template_handle/create_default_prompt.py
"""

from pathlib import Path

from ..util.path_utils import get_project_root, path_to_dot_notation
from ..util.template_utils import find_leaf_directories, get_template_base_dir


def generate_prompt_content(agent_name: str) -> str:
    """
    default.prompt.mdのファイル内容を生成する

    Args:
        agent_name: agent名（例: general.basic）

    Returns:
        str: ファイルに書き込む内容
    """
    content = f"""
    ---
    agent: {agent_name}
    ---    
    read .github/tasks/{agent_name}.md to understand your task.    
    """
    return content


def create_default_prompt_files(
    project_root: Path, dry_run: bool = False
) -> list[Path]:
    """
    全ての最下層ディレクトリにdefault.prompt.mdを作成する

    Args:
        project_root: プロジェクトルートディレクトリ
        dry_run: Trueの場合、実際にファイルを作成せずに対象を表示のみ

    Returns:
        list[Path]: 作成（または作成予定）のファイルパスリスト
    """
    template_base = get_template_base_dir(project_root)
    created_files: list[Path] = []

    leaf_dirs = find_leaf_directories(template_base)

    if not leaf_dirs:
        print("最下層ディレクトリが見つかりませんでした")
        return created_files

    print(f"検出された最下層ディレクトリ数: {len(leaf_dirs)}")
    print("-" * 60)

    for leaf_dir in leaf_dirs:
        agent_name = path_to_dot_notation(leaf_dir, template_base)
        prompt_file = leaf_dir / "default.prompt.md"
        content = generate_prompt_content(agent_name)

        if dry_run:
            print(f"[DRY RUN] 作成予定: {prompt_file}")
            print(f"  agent名: {agent_name}")
        else:
            # ファイルを作成（既存の場合は上書き）
            prompt_file.write_text(content, encoding="utf-8")
            print(f"作成完了: {prompt_file}")
            print(f"  agent名: {agent_name}")

        created_files.append(prompt_file)

    print("-" * 60)
    print(f"合計: {len(created_files)} ファイル")

    return created_files


def main() -> None:
    """
    メイン処理: スクリプトのエントリーポイント
    """
    project_root = get_project_root()
    print(f"プロジェクトルート: {project_root}")
    print("=" * 60)

    # 実際にファイルを作成
    # dry_run=Trueにすると、ファイルを作成せずに確認のみ
    create_default_prompt_files(project_root, dry_run=False)


if __name__ == "__main__":
    main()
