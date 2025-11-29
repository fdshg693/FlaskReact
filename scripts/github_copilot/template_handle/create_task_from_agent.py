"""
.github/agents配下の.agent.mdから.github/tasks/にtaskファイルを自動生成するスクリプト

==============================================================================
概要
==============================================================================
このスクリプトは、.github/agents/配下の{agent-name}.agent.mdファイルを探索し、
対応するtaskファイルを.github/tasks/配下に作成します。

==============================================================================
生成されるファイルの例
==============================================================================
入力:
    .github/agents/general.basic.agent.md
    .github/agents/coder.script.agent.md

出力:
    .github/tasks/general.basic.md
    .github/tasks/coder.script.md

==============================================================================
taskファイルの内容
==============================================================================
各taskファイルには、対応するエージェント用のタスクファイルである旨が記載されます。
例（general.basic.md）:
    general.basicエージェント用のタスクファイルです。

==============================================================================
使用方法
==============================================================================
    python scripts/github_copilot/template_handle/create_task_from_agent.py
"""

from util.path_utils import get_project_root
from util.template_utils import (
    get_agents_dir,
    get_tasks_dir,
)
from pathlib import Path


def find_agent_files_in_agents_dir(agents_dir: Path) -> list[Path]:
    """
    .github/agents配下の全ての.agent.mdファイルを取得する

    Args:
        agents_dir: .github/agentsディレクトリのパス

    Returns:
        list[Path]: .agent.mdファイルのパスリスト
    """
    agent_files: list[Path] = []

    if not agents_dir.exists():
        print(f"警告: ディレクトリが存在しません: {agents_dir}")
        return agent_files

    # {agent-name}.agent.md形式のファイルを探索
    for agent_file in agents_dir.glob("*.agent.md"):
        agent_files.append(agent_file)

    return agent_files


def extract_agent_name(agent_file: Path) -> str:
    """
    ファイル名からagent名を抽出する

    Args:
        agent_file: .agent.mdファイルのパス（例: general.basic.agent.md）

    Returns:
        str: agent名（例: general.basic）
    """
    # ファイル名から.agent.mdを除去してagent名を取得
    filename = agent_file.name
    return filename.removesuffix(".agent.md")


def generate_task_content(agent_name: str) -> str:
    """
    taskファイルの内容を生成する

    Args:
        agent_name: agent名（例: general.basic）

    Returns:
        str: ファイルに書き込む内容
    """
    return f"{agent_name}エージェント用のタスクファイルです。\n"


def create_task_files(project_root: Path, dry_run: bool = False) -> list[Path]:
    """
    全ての.agent.mdから対応するtaskファイルを作成する

    Args:
        project_root: プロジェクトルートディレクトリ
        dry_run: Trueの場合、実際にファイルを作成せずに対象を表示のみ

    Returns:
        list[Path]: 作成（または作成予定）のファイルパスリスト
    """
    agents_dir = get_agents_dir(project_root)
    tasks_dir = get_tasks_dir(project_root)
    created_files: list[Path] = []

    # tasksディレクトリが存在しない場合は作成
    if not dry_run and not tasks_dir.exists():
        tasks_dir.mkdir(parents=True, exist_ok=True)
        print(f"ディレクトリを作成しました: {tasks_dir}")

    agent_files = find_agent_files_in_agents_dir(agents_dir)

    if not agent_files:
        print(".agent.mdファイルが見つかりませんでした")
        return created_files

    print(f"検出された.agent.mdファイル数: {len(agent_files)}")
    print("-" * 60)

    for agent_file in agent_files:
        # ファイル名からagent名を抽出（例: general.basic.agent.md -> general.basic）
        agent_name = extract_agent_name(agent_file)
        task_file = tasks_dir / f"{agent_name}.md"
        content = generate_task_content(agent_name)

        if dry_run:
            print(f"[DRY RUN] 作成予定: {task_file}")
            print(f"  元ファイル: {agent_file}")
        else:
            # ファイルを作成（既存の場合は上書き）
            task_file.write_text(content, encoding="utf-8")
            print(f"作成完了: {task_file}")
            print(f"  元ファイル: {agent_file}")

        created_files.append(task_file)

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
    create_task_files(project_root, dry_run=False)


if __name__ == "__main__":
    main()
