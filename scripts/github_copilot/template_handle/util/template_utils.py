"""
.github_copilot_template関連のユーティリティ関数

テンプレートディレクトリの探索や、ファイル検出などの共通処理を提供します。
"""

from pathlib import Path


def get_template_base_dir(project_root: Path) -> Path:
    """
    .github_copilot_templateディレクトリのパスを取得する

    Args:
        project_root: プロジェクトルートのパス

    Returns:
        Path: .github_copilot_templateディレクトリのパス
    """
    return project_root / ".github_copilot_template"


def get_tasks_dir(project_root: Path) -> Path:
    """
    .github/tasksディレクトリのパスを取得する

    Args:
        project_root: プロジェクトルートのパス

    Returns:
        Path: .github/tasksディレクトリのパス
    """
    return project_root / ".github" / "tasks"


def get_agents_dir(project_root: Path) -> Path:
    """
    .github/agentsディレクトリのパスを取得する

    Args:
        project_root: プロジェクトルートのパス

    Returns:
        Path: .github/agentsディレクトリのパス
    """
    return project_root / ".github" / "agents"


def get_prompts_dir(project_root: Path) -> Path:
    """
    .github/promptsディレクトリのパスを取得する

    Args:
        project_root: プロジェクトルートのパス

    Returns:
        Path: .github/promptsディレクトリのパス
    """
    return project_root / ".github" / "prompts"


def find_leaf_directories(base_dir: Path) -> list[Path]:
    """
    指定ディレクトリ配下の最下層ディレクトリを全て取得する

    最下層ディレクトリとは、子ディレクトリを持たないディレクトリのことです。

    Args:
        base_dir: 探索対象のベースディレクトリ

    Returns:
        list[Path]: 最下層ディレクトリのパスリスト
    """
    leaf_dirs: list[Path] = []

    if not base_dir.exists():
        print(f"警告: ディレクトリが存在しません: {base_dir}")
        return leaf_dirs

    for dir_path in base_dir.rglob("*"):
        if dir_path.is_dir():
            # 子ディレクトリを持たないディレクトリを最下層とみなす
            subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
            if not subdirs:
                leaf_dirs.append(dir_path)

    return leaf_dirs


def find_agent_files(base_dir: Path) -> list[Path]:
    """
    指定ディレクトリ配下の全ての.agent.mdファイルを取得する

    Args:
        base_dir: 探索対象のベースディレクトリ

    Returns:
        list[Path]: .agent.mdファイルのパスリスト
    """
    agent_files: list[Path] = []

    if not base_dir.exists():
        print(f"警告: ディレクトリが存在しません: {base_dir}")
        return agent_files

    # 再帰的に.agent.mdファイルを探索
    for agent_file in base_dir.rglob(".agent.md"):
        agent_files.append(agent_file)

    return agent_files


def find_prompt_files(base_dir: Path) -> list[Path]:
    """
    指定ディレクトリ配下の全ての.prompt.mdファイルを取得する

    Args:
        base_dir: 探索対象のベースディレクトリ

    Returns:
        list[Path]: .prompt.mdファイルのパスリスト
    """
    prompt_files: list[Path] = []

    if not base_dir.exists():
        print(f"警告: ディレクトリが存在しません: {base_dir}")
        return prompt_files

    # 再帰的に.prompt.mdファイルを探索
    for prompt_file in base_dir.rglob("*.prompt.md"):
        prompt_files.append(prompt_file)

    return prompt_files
