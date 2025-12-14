"""
deploy_agents.py および関連ユーティリティのテスト

テスト対象:
- yaml_utils.py: フロントマター解析
- substitution_utils.py: 変数置換
- deploy_agents.py: エージェントファイルのデプロイ
"""

import sys
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from github_template.deployers.deploy_agents import (  # noqa: E402
    deploy_agents,
    filter_by_patterns,
    generate_dest_filename,
    process_agent_file,
)
from github_template.util.substitution_utils import (  # noqa: E402
    get_output_values_without_name,
    substitute_custom_variables,
)
from github_template.util.yaml_utils import (  # noqa: E402
    extract_custom_inputs,
    extract_outputs,
    parse_frontmatter,
    rebuild_content_with_frontmatter,
    remove_custom_sections_from_frontmatter,
)


class TestParseFrontmatter:
    """parse_frontmatter関数のテスト"""

    def test_valid_frontmatter(self):
        """正常なフロントマターの解析"""
        content = """---
description: テスト説明
tools: ['search', 'edit']
custom_inputs:
  - name: target
    type: string
    default: "default_value"
outputs:
  - name: default
    target: default
---
本文の内容
"""
        frontmatter, body = parse_frontmatter(content)

        assert frontmatter["description"] == "テスト説明"
        assert frontmatter["tools"] == ["search", "edit"]
        assert len(frontmatter["custom_inputs"]) == 1
        assert frontmatter["custom_inputs"][0]["name"] == "target"
        assert len(frontmatter["outputs"]) == 1
        assert "本文の内容" in body

    def test_missing_frontmatter(self):
        """フロントマターがない場合はエラー"""
        content = "本文のみ"
        with pytest.raises(ValueError, match="フロントマターが見つかりません"):
            parse_frontmatter(content)

    def test_unclosed_frontmatter(self):
        """フロントマターが閉じられていない場合はエラー"""
        content = """---
description: テスト
本文"""
        with pytest.raises(ValueError, match="フロントマターの終了が見つかりません"):
            parse_frontmatter(content)


class TestExtractCustomInputs:
    """extract_custom_inputs関数のテスト"""

    def test_extract_single_input(self):
        """単一のcustom_inputを抽出"""
        frontmatter = {
            "custom_inputs": [
                {"name": "target", "type": "string", "default": "default_value"}
            ]
        }
        result = extract_custom_inputs(frontmatter)
        assert result == {"target": "default_value"}

    def test_extract_multiple_inputs(self):
        """複数のcustom_inputsを抽出"""
        frontmatter = {
            "custom_inputs": [
                {"name": "target", "default": "path/to"},
                {"name": "language", "default": "japanese"},
            ]
        }
        result = extract_custom_inputs(frontmatter)
        assert result == {"target": "path/to", "language": "japanese"}

    def test_missing_custom_inputs(self):
        """custom_inputsがない場合は空の辞書"""
        frontmatter = {"description": "test"}
        result = extract_custom_inputs(frontmatter)
        assert result == {}

    def test_missing_default(self):
        """defaultがない場合は空文字"""
        frontmatter = {"custom_inputs": [{"name": "target"}]}
        result = extract_custom_inputs(frontmatter)
        assert result == {"target": ""}


class TestExtractOutputs:
    """extract_outputs関数のテスト"""

    def test_extract_outputs(self):
        """outputsセクションを抽出"""
        frontmatter = {
            "outputs": [
                {"name": "default", "target": "default"},
                {"name": "custom", "target": "src/"},
            ]
        }
        result = extract_outputs(frontmatter)
        assert len(result) == 2
        assert result[0]["name"] == "default"
        assert result[1]["name"] == "custom"

    def test_missing_outputs(self):
        """outputsがない場合はエラー"""
        frontmatter = {"description": "test"}
        with pytest.raises(ValueError, match="outputsセクションが存在しません"):
            extract_outputs(frontmatter)

    def test_empty_outputs(self):
        """outputsが空の場合はエラー"""
        frontmatter = {"outputs": []}
        with pytest.raises(ValueError, match="outputsセクションが存在しません"):
            extract_outputs(frontmatter)


class TestRemoveCustomSections:
    """remove_custom_sections_from_frontmatter関数のテスト"""

    def test_remove_sections(self):
        """custom_inputsとoutputsを削除"""
        frontmatter = {
            "description": "test",
            "tools": ["search"],
            "custom_inputs": [{"name": "target"}],
            "outputs": [{"name": "default"}],
        }
        result = remove_custom_sections_from_frontmatter(frontmatter)
        assert "description" in result
        assert "tools" in result
        assert "custom_inputs" not in result
        assert "outputs" not in result

    def test_no_sections_to_remove(self):
        """削除対象がない場合はそのまま"""
        frontmatter = {"description": "test"}
        result = remove_custom_sections_from_frontmatter(frontmatter)
        assert result == {"description": "test"}


class TestRebuildContent:
    """rebuild_content_with_frontmatter関数のテスト"""

    def test_rebuild_with_frontmatter(self):
        """フロントマターと本文を結合"""
        frontmatter = {"description": "test"}
        body = "\n本文"
        result = rebuild_content_with_frontmatter(frontmatter, body)
        assert result.startswith("---\n")
        assert "description: test" in result
        assert "---\n\n本文" in result

    def test_rebuild_with_tools_list(self):
        """toolsリストがフロースタイルで保持される"""
        frontmatter = {
            "description": "test",
            "tools": ["edit", "search", "runCommands"],
        }
        body = "\n本文"
        result = rebuild_content_with_frontmatter(frontmatter, body)
        # リストがフロースタイル（インライン）で出力されることを確認
        assert "[edit, search, runCommands]" in result

    def test_empty_frontmatter(self):
        """フロントマターが空の場合は本文のみ"""
        frontmatter = {}
        body = "\n\n本文"
        result = rebuild_content_with_frontmatter(frontmatter, body)
        assert result == "本文"


class TestSubstituteCustomVariables:
    """substitute_custom_variables関数のテスト"""

    def test_substitute_with_default(self):
        """defaultの場合は${input:name:"default_value"}に変換"""
        content = "ターゲット: ${custom:target}"
        output_values = {"target": "default"}
        custom_inputs = {"target": "ルートディレクトリ"}

        result = substitute_custom_variables(content, output_values, custom_inputs)
        assert result == 'ターゲット: ${input:target:"ルートディレクトリ"}'

    def test_substitute_with_value(self):
        """default以外の場合は値で直接置換"""
        content = "ターゲット: ${custom:target}"
        output_values = {"target": "src/"}
        custom_inputs = {"target": "ルートディレクトリ"}

        result = substitute_custom_variables(content, output_values, custom_inputs)
        assert result == "ターゲット: src/"

    def test_multiple_variables(self):
        """複数の変数を置換"""
        content = "パス: ${custom:target}, 言語: ${custom:language}"
        output_values = {"target": "default", "language": "japanese"}
        custom_inputs = {"target": "root", "language": "english"}

        result = substitute_custom_variables(content, output_values, custom_inputs)
        assert '${input:target:"root"}' in result
        assert "japanese" in result

    def test_unknown_variable(self):
        """不明な変数はそのまま残す"""
        content = "${custom:unknown}"
        output_values = {}
        custom_inputs = {}

        result = substitute_custom_variables(content, output_values, custom_inputs)
        assert result == "${custom:unknown}"


class TestGetOutputValuesWithoutName:
    """get_output_values_without_name関数のテスト"""

    def test_remove_name(self):
        """nameを除いた値を取得"""
        output_entry = {"name": "default", "target": "src/", "language": "jp"}
        result = get_output_values_without_name(output_entry)
        assert "name" not in result
        assert result["target"] == "src/"
        assert result["language"] == "jp"


class TestGenerateDestFilename:
    """generate_dest_filename関数のテスト"""

    def test_generate_filename(self):
        """ファイル名を正しく生成"""
        template_dir = Path("/project/.github_copilot_template")
        agent_file = template_dir / "docs" / "ai_knowledge" / ".agent.md"
        output_name = "default"

        result = generate_dest_filename(agent_file, template_dir, output_name)
        assert result == "docs.ai_knowledge.default.agent.md"

    def test_generate_filename_with_custom_name(self):
        """カスタム名でファイル名を生成"""
        template_dir = Path("/project/.github_copilot_template")
        agent_file = template_dir / "coder" / "script" / ".agent.md"
        output_name = "japanese"

        result = generate_dest_filename(agent_file, template_dir, output_name)
        assert result == "coder.script.japanese.agent.md"


class TestFilterByPatterns:
    """filter_by_patterns関数のテスト"""

    def test_filter_by_directory_pattern(self):
        """ディレクトリパターンでフィルタリング"""
        template_dir = Path("/project/.github_copilot_template")
        agent_files = [
            template_dir / "coder" / "script" / ".agent.md",
            template_dir / "coder" / "review" / ".agent.md",
            template_dir / "docs" / "readme" / ".agent.md",
        ]
        patterns = ["coder/"]

        result = filter_by_patterns(agent_files, patterns, template_dir)
        assert len(result) == 2
        assert all("coder" in str(f) for f in result)

    def test_filter_by_file_pattern(self):
        """ファイルパターンでフィルタリング"""
        template_dir = Path("/project/.github_copilot_template")
        agent_files = [
            template_dir / "coder" / "script" / ".agent.md",
            template_dir / "docs" / "readme" / ".agent.md",
        ]
        patterns = ["docs/readme"]

        result = filter_by_patterns(agent_files, patterns, template_dir)
        assert len(result) == 1
        assert "readme" in str(result[0])


class TestProcessAgentFile:
    """process_agent_file関数のテスト"""

    def test_process_valid_file(self, tmp_path: Path) -> None:
        """正常なファイルの処理"""
        # テンプレートディレクトリを作成
        template_dir = tmp_path / ".github_copilot_template"
        agent_dir = template_dir / "test"
        agent_dir.mkdir(parents=True)

        # テストファイルを作成
        agent_file = agent_dir / ".agent.md"
        agent_file.write_text(
            """---
description: テスト
custom_inputs:
  - name: target
    default: "default_path"
outputs:
  - name: default
    target: default
  - name: custom
    target: src/
---
ターゲット: ${custom:target}
""",
            encoding="utf-8",
        )

        # 宛先ディレクトリを作成
        dest_dir = tmp_path / ".github" / "agents"
        dest_dir.mkdir(parents=True)

        # 処理を実行
        copied, skipped, errors = process_agent_file(
            agent_file, template_dir, dest_dir, overwrite=True
        )

        # 結果を検証
        assert len(copied) == 2
        assert len(errors) == 0

        # 生成されたファイルを確認
        default_file = dest_dir / "test.default.agent.md"
        custom_file = dest_dir / "test.custom.agent.md"
        assert default_file.exists()
        assert custom_file.exists()

        # 内容を確認
        default_content = default_file.read_text(encoding="utf-8")
        custom_content = custom_file.read_text(encoding="utf-8")

        assert '${input:target:"default_path"}' in default_content
        assert "src/" in custom_content
        assert "custom_inputs" not in default_content
        assert "outputs" not in custom_content

    def test_process_file_without_outputs(self, tmp_path: Path) -> None:
        """outputsがないファイルはエラー"""
        template_dir = tmp_path / ".github_copilot_template"
        agent_dir = template_dir / "test"
        agent_dir.mkdir(parents=True)

        agent_file = agent_dir / ".agent.md"
        agent_file.write_text(
            """---
description: テスト
---
本文
""",
            encoding="utf-8",
        )

        dest_dir = tmp_path / ".github" / "agents"
        dest_dir.mkdir(parents=True)

        copied, skipped, errors = process_agent_file(
            agent_file, template_dir, dest_dir, overwrite=True
        )

        assert len(errors) == 1
        assert "outputsセクションが存在しません" in errors[0]


class TestDeployAgents:
    """deploy_agents関数のテスト"""

    def test_deploy_with_clean(self, tmp_path: Path) -> None:
        """クリーンモードでデプロイ"""
        # テンプレートを作成
        template_dir = tmp_path / ".github_copilot_template"
        agent_dir = template_dir / "test"
        agent_dir.mkdir(parents=True)

        agent_file = agent_dir / ".agent.md"
        agent_file.write_text(
            """---
description: テスト
outputs:
  - name: default
---
本文
""",
            encoding="utf-8",
        )

        # 宛先に既存ファイルを作成
        dest_dir = tmp_path / ".github" / "agents"
        dest_dir.mkdir(parents=True)
        old_file = dest_dir / "old.agent.md"
        old_file.write_text("古いファイル")

        # クリーンモードでデプロイ
        copied, skipped, errors, deleted = deploy_agents(
            template_dir, dest_dir, overwrite=True, clean=True
        )

        assert len(deleted) == 1
        assert str(old_file) in deleted[0]
        assert len(copied) == 1

    def test_deploy_no_overwrite(self, tmp_path: Path) -> None:
        """上書きなしモードでデプロイ"""
        template_dir = tmp_path / ".github_copilot_template"
        agent_dir = template_dir / "test"
        agent_dir.mkdir(parents=True)

        agent_file = agent_dir / ".agent.md"
        agent_file.write_text(
            """---
outputs:
  - name: default
---
本文
""",
            encoding="utf-8",
        )

        dest_dir = tmp_path / ".github" / "agents"
        dest_dir.mkdir(parents=True)

        # 既存ファイルを作成
        existing = dest_dir / "test.default.agent.md"
        existing.write_text("既存")

        copied, skipped, errors, deleted = deploy_agents(
            template_dir, dest_dir, overwrite=False
        )

        assert len(skipped) == 1
        assert len(copied) == 0
