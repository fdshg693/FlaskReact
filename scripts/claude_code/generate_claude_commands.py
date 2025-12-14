"""
Claude CLI Command Generator

Reads YAML configuration files and generates corresponding Claude CLI commands,
automatically updating the 'command' field in each configuration file.
"""

import argparse
import glob
import logging
import re
import sys
from pathlib import Path
from typing import Final, TypeAlias, cast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Model shorthand to full name mapping
MODEL_MAP: Final[dict[str, str]] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-5",
    "opus": "claude-opus-4-5",
}

# All available Claude tools
ALL_TOOLS: Final[list[str]] = [
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "KillShell",
    "NotebookEdit",
    "Read",
    "Skill",
    "SlashCommand",
    "Task",
    "Write",
    "WebFetch",
    "WebSearch",
]

YamlList: TypeAlias = list[str]
YamlValue: TypeAlias = str | int | YamlList
YamlDict: TypeAlias = dict[str, YamlValue]

EMPTY_STR_LIST: Final[list[str]] = []


def parse_yaml_with_frontmatter(content: str) -> tuple[YamlDict, YamlDict]:
    """
    Parse YAML content with front matter format (two --- delimited sections).

    Returns:
        Tuple of (config_section, output_section)
    """
    # Split by '---' delimiter
    parts = re.split(r"^---\s*$", content, flags=re.MULTILINE)

    # Filter out empty parts
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        raise ValueError("Invalid YAML format: expected two sections separated by ---")

    config_section = parse_simple_yaml(parts[0])
    output_section = parse_simple_yaml(parts[1])

    return config_section, output_section


def parse_simple_yaml(yaml_str: str) -> YamlDict:
    """
    Parse simple YAML without external dependencies.
    Supports: strings, numbers, lists, and nested keys.
    """
    result: YamlDict = {}
    current_list_key: str | None = None
    lines = yaml_str.split("\n")

    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.strip().startswith("#"):
            continue

        # Check for list item
        list_match = re.match(r"^(\s*)-\s*(.+)$", line)
        if list_match and current_list_key:
            value = list_match.group(2).strip()
            value = value.strip('"').strip("'")
            cast(list[str], result[current_list_key]).append(value)
            continue

        # Check for key-value pair
        kv_match = re.match(r"^(\w+):\s*(.*)$", line)
        if kv_match:
            key = kv_match.group(1)
            value = kv_match.group(2).strip()

            if value == "":
                # Could be a list starting on next line
                current_list_key = key
                result[key] = cast(list[str], [])
            elif value.startswith('"') and value.endswith('"'):
                result[key] = value[1:-1]
                current_list_key = None
            elif value.startswith("'") and value.endswith("'"):
                result[key] = value[1:-1]
                current_list_key = None
            elif value.isdigit():
                result[key] = int(value)
                current_list_key = None
            else:
                result[key] = value
                current_list_key = None

    return result


def serialize_yaml_with_frontmatter(config: YamlDict, output: YamlDict) -> str:
    """
    Serialize config and output sections back to YAML with front matter format.
    """

    def serialize_value(value: YamlValue, indent: int = 0) -> str:
        if isinstance(value, list):
            if not value:
                return ""
            lines: list[str] = []
            for item in cast(list[str], value):
                lines.append(f'{"  " * indent}  - "{item}"')
            return "\n".join(lines)
        elif isinstance(value, str):
            if value == "":
                return '""'
            # Escape and quote if needed
            if " " in value or '"' in value or any(c in value for c in ":{}[]"):
                return f'"{value}"'
            return f'"{value}"'
        elif isinstance(value, int):
            return str(value)
        return str(value)

    lines = ["---"]

    # Serialize config section
    for key, value in config.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f'  - "{item}"')
        else:
            lines.append(f"{key}: {serialize_value(value)}")

    lines.append("---")

    # Serialize output section
    for key, value in output.items():
        lines.append(f"{key}: {serialize_value(value)}")

    return "\n".join(lines) + "\n"


def generate_command(config: YamlDict) -> tuple[str, str]:
    """
    Generate Claude CLI command from configuration.

    Returns:
        Tuple of (command, error_message)
    """
    errors: list[str] = []

    # Validate required fields
    if "name" not in config:
        errors.append("no name specified")

    if "model" not in config:
        errors.append("no model specified")

    if errors:
        return "", "; ".join(errors)

    # Build command parts
    cmd_parts: list[str] = ["claude", "-p"]

    # Model mapping
    model_short = cast(str, config["model"])
    if model_short not in MODEL_MAP:
        return (
            "",
            f"invalid model '{model_short}' - must be one of: haiku, sonnet, opus",
        )

    cmd_parts.extend(["--model", MODEL_MAP[model_short]])

    # System prompt handling
    system_prompt = cast(str, config.get("system_prompt", "")).strip()
    system_prompt_file = cast(str, config.get("system_prompt_file", "")).strip()

    if system_prompt:
        # Inline system prompt takes priority
        # For inline, we'd need --system-prompt, but typically file is used
        # Based on the task, it seems file-based is the main approach
        cmd_parts.extend(["--system-prompt", f'"{system_prompt}"'])
    elif system_prompt_file:
        cmd_parts.extend(["--system-prompt-file", system_prompt_file])
    else:
        logger.warning(
            f"No system prompt specified for '{config.get('name', 'unknown')}'"
        )

    # Tool permissions
    allowed_tools = cast(list[str], config.get("allowed_tools", EMPTY_STR_LIST))

    if not allowed_tools:
        # No tools allowed - disallow all
        disallowed_tools = ALL_TOOLS.copy()
    else:
        # Validate allowed tools
        valid_allowed: list[str] = []
        for tool in allowed_tools:
            if tool in ALL_TOOLS:
                valid_allowed.append(tool)
            else:
                logger.warning(f"Invalid tool '{tool}' in allowed_tools - ignoring")

        disallowed_tools = [t for t in ALL_TOOLS if t not in valid_allowed]

    if disallowed_tools:
        cmd_parts.append("--disallowedTools")
        for tool in disallowed_tools:
            cmd_parts.append(f'"{tool}"')

    # Max turns
    max_turns = cast(int, config.get("max_turns", 5))
    cmd_parts.extend(["--max-turns", str(max_turns)])

    return " ".join(cmd_parts), ""


def process_file(file_path: Path) -> bool:
    """
    Process a single YAML configuration file.

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"{file_path} - Failed to read file: {e}")
        return False

    try:
        config, output = parse_yaml_with_frontmatter(content)
    except ValueError as e:
        logger.error(f"{file_path} - {e}")
        return False

    # Generate command
    command, error = generate_command(config)

    # Update output section
    if error:
        output["command"] = ""
        output["error"] = error
        logger.error(f"{file_path.name} - {error}")
    else:
        output["command"] = command
        output["error"] = ""
        logger.info(f"{file_path.name} - Command generated successfully")

    # Write back to file
    try:
        new_content = serialize_yaml_with_frontmatter(config, output)
        file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        logger.error(f"{file_path} - Failed to write file: {e}")
        return False

    return error == ""


def resolve_paths(path_input: str) -> list[Path]:
    """
    Resolve comma-separated paths to list of YAML files.
    Handles files, directories, and glob patterns.
    """
    files: list[Path] = []
    paths: list[str] = [p.strip() for p in path_input.split(",")]

    for path_str in paths:
        if not path_str:
            continue

        path = Path(path_str)

        # Check if it's a glob pattern
        if "*" in path_str or "?" in path_str:
            matched = glob.glob(path_str, recursive=True)
            for m in matched:
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() in [".yaml", ".yml"]:
                    files.append(mp)
        elif path.is_file():
            if path.suffix.lower() in [".yaml", ".yml"]:
                files.append(path)
            else:
                logger.warning(f"Skipping non-YAML file: {path}")
        elif path.is_dir():
            # Find all YAML files in directory
            for ext in ["*.yaml", "*.yml"]:
                files.extend(path.glob(ext))
        else:
            logger.warning(f"Path does not exist: {path}")

    return files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Claude CLI commands from YAML configuration files."
    )
    parser.add_argument(
        "paths",
        help="Comma-separated list of paths (files, directories, or glob patterns)",
    )

    args = parser.parse_args()

    files = resolve_paths(args.paths)

    if not files:
        logger.error("No YAML files found to process")
        sys.exit(1)

    logger.info(f"Found {len(files)} file(s) to process")

    success_count = 0
    error_count = 0

    for file_path in files:
        if process_file(file_path):
            success_count += 1
        else:
            error_count += 1

    # Summary
    print()
    print("=" * 50)
    print(f"Summary: {success_count} succeeded, {error_count} failed")
    print("=" * 50)

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
