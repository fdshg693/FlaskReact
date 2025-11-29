"""
`.github_copilot_template\agents`を元に、.github\agentsへエージェントを展開するスクリプト。
Agent Template Deployment Script

Copies agent files from .github_copilot_template/agents/ to .github/agents/,
flattening the directory structure by converting path separators to dots.

Usage:
    python scripts/github_copilot/setting_utils/deploy_agents.py [config.yaml] [--overwrite | --no-overwrite]

Arguments:
    config.yaml     Optional YAML config file specifying which agents to include
    --overwrite     Overwrite existing agent files (default)
    --no-overwrite  Skip files that already exist in destination

Examples:
    python scripts/github_copilot/setting_utils/deploy_agents.py                              # Copy all agents, overwrite existing
    python scripts/github_copilot/setting_utils/deploy_agents.py agents-config.yaml           # Copy only specified agents
    python scripts/github_copilot/setting_utils/deploy_agents.py --no-overwrite               # Copy all, skip existing files
    python scripts/github_copilot/setting_utils/deploy_agents.py config.yaml --no-overwrite   # Copy specified, skip existing
"""

import os
import shutil
import sys
from pathlib import Path


def load_yaml_config(config_path: str) -> list[str]:
    """
    Parse a simple YAML config file for include patterns.
    Returns a list of include patterns.

    Note: Uses simple parsing to avoid external dependencies.
    """
    includes = []
    with open(config_path, "r", encoding="utf-8") as f:
        in_include_section = False
        for line in f:
            stripped = line.strip()
            if stripped == "include:":
                in_include_section = True
                continue
            if in_include_section:
                if stripped.startswith("- "):
                    # Extract the pattern, remove comments
                    pattern = stripped[2:].split("#")[0].strip()
                    if pattern:
                        includes.append(pattern)
                elif stripped and not stripped.startswith("#"):
                    # New section started
                    break
    return includes


def should_include_file(relative_path: str, includes: list[str]) -> bool:
    """
    Check if a file should be included based on the include patterns.

    Args:
        relative_path: Path relative to .github_copilot_template/agents/ (e.g., 'coder/backend.agent.md')
        includes: List of include patterns from YAML config

    Returns:
        True if file should be included
    """
    if not includes:
        return True  # No config means include all

    for pattern in includes:
        if pattern.endswith("/"):
            # Directory pattern - check if file is under this directory
            dir_pattern = pattern.rstrip("/")
            if relative_path.startswith(dir_pattern + "/") or relative_path.startswith(
                dir_pattern + "\\"
            ):
                return True
        else:
            # File pattern - check for exact match (without extension)
            file_name_without_ext = os.path.splitext(os.path.basename(relative_path))[0]
            # Remove .agent suffix if present
            if file_name_without_ext.endswith(".agent"):
                file_name_without_ext = file_name_without_ext[:-6]

            # Check if pattern matches the file name or relative path without extension
            path_without_ext = os.path.splitext(relative_path)[0]
            if path_without_ext.endswith(".agent"):
                path_without_ext = path_without_ext[:-6]

            if pattern == file_name_without_ext or pattern == path_without_ext:
                return True

    return False


def flatten_path(relative_path: str) -> str:
    """
    Convert a relative path to a flattened filename.
    Path separators are converted to dots.

    Example: 'coder/backend.agent.md' -> 'coder.backend.agent.md'
    """
    # Normalize path separators and convert to dots
    normalized = relative_path.replace("\\", "/").replace("/", ".")
    return normalized


def deploy_agents(
    source_dir: Path, dest_dir: Path, includes: list[str] = None, overwrite: bool = True
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Deploy agent files from source to destination with flattened structure.

    Args:
        source_dir: Source directory (.github_copilot_template/agents/)
        dest_dir: Destination directory (.github/agents/)
        includes: Optional list of include patterns
        overwrite: If True, overwrite existing files. If False, skip them.

    Returns:
        Tuple of (copied_files, skipped_files) where:
        - copied_files: List of (source_path, dest_path) tuples for copied files
        - skipped_files: List of destination paths that were skipped (already exist)
    """
    if includes is None:
        includes = []

    copied_files = []
    skipped_files = []

    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Walk through source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_path = Path(root) / file
            relative_path = source_path.relative_to(source_dir).as_posix()

            # Check if file should be included
            if not should_include_file(relative_path, includes):
                continue

            # Create flattened destination filename
            dest_filename = flatten_path(relative_path)
            dest_path = dest_dir / dest_filename

            # Check if file exists and handle overwrite option
            if dest_path.exists() and not overwrite:
                skipped_files.append(str(dest_path))
                print(f"Skipped (exists): {relative_path} -> {dest_filename}")
                continue

            # Copy file
            shutil.copy2(source_path, dest_path)
            copied_files.append((str(source_path), str(dest_path)))
            print(f"Copied: {relative_path} -> {dest_filename}")

    return copied_files, skipped_files


def parse_args(args: list[str]) -> tuple[str | None, bool]:
    """
    Parse command line arguments.

    Args:
        args: List of command line arguments (excluding script name)

    Returns:
        Tuple of (config_path, overwrite_flag)
    """
    config_path = None
    overwrite = True  # Default: overwrite existing files

    for arg in args:
        if arg == "--overwrite":
            overwrite = True
        elif arg == "--no-overwrite":
            overwrite = False
        elif not arg.startswith("-"):
            config_path = arg

    return config_path, overwrite


def main():
    # Determine project root (parent of scripts folder)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent

    source_dir = project_root / ".github_copilot_template" / "agents"
    dest_dir = project_root / ".github" / "agents"

    # Check source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Parse command line arguments
    config_path, overwrite = parse_args(sys.argv[1:])

    # Load config if provided
    includes = []
    if config_path:
        if not os.path.exists(config_path):
            # Try relative to project root
            config_path = project_root / config_path
        if os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            includes = load_yaml_config(str(config_path))
            print(f"Include patterns: {includes}")
        else:
            print(f"Warning: Config file not found: {sys.argv[1]}")

    # Deploy agents
    print("\nDeploying agents...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Overwrite existing: {overwrite}\n")

    copied, skipped = deploy_agents(source_dir, dest_dir, includes, overwrite)

    print("\nDeployment complete.")
    print(f"  Copied: {len(copied)} file(s)")
    if skipped:
        print(f"  Skipped: {len(skipped)} file(s) (already exist)")


if __name__ == "__main__":
    main()
