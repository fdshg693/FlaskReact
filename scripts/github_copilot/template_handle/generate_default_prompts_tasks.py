"""
`.github/agents`を元に、デフォルトのプロンプトテンプレートファイルおよび、タスクファイルを生成するためにユーザが実行するスクリプト。
Generate default prompt files and task files based on existing agent configuration files.

Scans .github/agents/ folder for agent files and creates corresponding:
- default.prompt.md files in .github/prompts/ folder
- task files in .github/tasks/ folder (only if not already exists)
"""

from pathlib import Path


def find_agent_files(agents_dir: Path) -> list[Path]:
    """Find all .agent.md files recursively in the agents directory."""
    return list(agents_dir.rglob("*.agent.md"))


def get_agent_name(agent_file: Path, agents_dir: Path) -> str:
    """
    Extract agent name from file path.

    Example:
        .github/agents/coder/backend.agent.md -> coder.backend
        .github/agents/general/basic.agent.md -> general.basic
    """
    relative = agent_file.relative_to(agents_dir)
    # Get parent directories and filename without .agent.md
    parts = list(relative.parent.parts) + [relative.stem.replace(".agent", "")]
    return ".".join(parts)


def get_prompt_path(agent_file: Path, agents_dir: Path, prompts_dir: Path) -> Path:
    """
    Calculate the prompt file path for an agent file.

    Example:
        .github/agents/coder/backend.agent.md -> .github/prompts/coder/backend/default.prompt.md
    """
    relative = agent_file.relative_to(agents_dir)
    # Remove .agent.md suffix and build prompt path
    agent_subpath = relative.with_suffix("").with_suffix("")  # Remove .agent.md
    return prompts_dir / agent_subpath / "default.prompt.md"


def get_task_path(agent_name: str, tasks_dir: Path) -> Path:
    """
    Calculate the task file path for an agent.

    Example:
        coder.backend -> .github/tasks/coder.backend.md
    """
    return tasks_dir / f"{agent_name}.md"


def create_prompt_content(agent_name: str) -> str:
    """Generate the content for a default prompt file."""
    return f"""---
agent: {agent_name}
---
Do your task written in `.github/tasks/{agent_name}.md`
"""


def create_task_content(agent_name: str) -> str:
    """Generate the content for a default task file."""
    return f"""# Task for {agent_name}

<!-- Write your task description here -->

"""


def generate_prompts(base_dir: Path) -> list[str]:
    """Generate all default prompt files and return list of created files."""
    agents_dir = base_dir / ".github" / "agents"
    prompts_dir = base_dir / ".github" / "prompts"

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        return []

    created_files = []
    agent_files = find_agent_files(agents_dir)

    for agent_file in agent_files:
        agent_name = get_agent_name(agent_file, agents_dir)
        prompt_path = get_prompt_path(agent_file, agents_dir, prompts_dir)
        content = create_prompt_content(agent_name)

        # Create parent directories if needed
        prompt_path.parent.mkdir(parents=True, exist_ok=True)

        # Write prompt file
        prompt_path.write_text(content, encoding="utf-8")
        created_files.append(str(prompt_path.relative_to(base_dir)))
        print(f"Created: {prompt_path.relative_to(base_dir)}")

    return created_files


def generate_tasks(base_dir: Path) -> tuple[list[str], list[str]]:
    """
    Generate task files for all agents and return lists of created and skipped files.

    Only creates task files if they don't already exist.
    """
    agents_dir = base_dir / ".github" / "agents"
    tasks_dir = base_dir / ".copilot" / "tasks"

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        return [], []

    created_files = []
    skipped_files = []
    agent_files = find_agent_files(agents_dir)

    for agent_file in agent_files:
        agent_name = get_agent_name(agent_file, agents_dir)
        task_path = get_task_path(agent_name, tasks_dir)

        # Skip if task file already exists
        if task_path.exists():
            skipped_files.append(str(task_path.relative_to(base_dir)))
            print(f"Skipped (already exists): {task_path.relative_to(base_dir)}")
            continue

        content = create_task_content(agent_name)

        # Create parent directories if needed
        task_path.parent.mkdir(parents=True, exist_ok=True)

        # Write task file
        task_path.write_text(content, encoding="utf-8")
        created_files.append(str(task_path.relative_to(base_dir)))
        print(f"Created: {task_path.relative_to(base_dir)}")

    return created_files, skipped_files


def main():
    # Get the project root directory (parent of scripts folder)
    script_dir = Path(__file__).parent
    base_dir = (
        script_dir.parent.parent.parent
    )  # Go up two levels: user -> scripts -> project root

    print(f"Base directory: {base_dir}")
    print("Scanning for agent files...")
    print()

    print("=== Generating Prompt Files ===")
    created_prompts = generate_prompts(base_dir)

    print()
    print("=== Generating Task Files ===")
    created_tasks, skipped_tasks = generate_tasks(base_dir)

    print()
    print("=== Summary ===")
    print(f"Prompt files created: {len(created_prompts)}")
    print(f"Task files created: {len(created_tasks)}")
    print(f"Task files skipped: {len(skipped_tasks)}")


if __name__ == "__main__":
    main()
