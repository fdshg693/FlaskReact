"""
ClaudeのSkillについて扱う勉強用スクリプト
https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart
"""

import anthropic

from config import load_dotenv_workspace

load_dotenv_workspace()
client = anthropic.Anthropic()


def show_skills():
    """
    Claudeがデフォルトで利用できるスキル一覧を表示する
    """
    # List Anthropic-managed Skills
    skills = client.beta.skills.list(source="anthropic", betas=["skills-2025-10-02"])

    for skill in skills.data:
        print(f"{skill.id}: {skill.display_title}")

    # xlsx: xlsx
    # pptx: pptx
    # pdf: pdf
    # docx: docx
