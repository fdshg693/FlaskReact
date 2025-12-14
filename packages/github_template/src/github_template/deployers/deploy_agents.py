"""
エージェントデプロイスクリプト

==============================================================================
概要
==============================================================================
`.github_copilot_template/` 配下の `.agent.md` ファイルを
`.github/agents/` へ展開するスクリプトです。
`.agent.md` 以外のファイルは無視されます。

==============================================================================
パス変換ルール
==============================================================================
テンプレートディレクトリからの相対パスを `.` 区切りに変換して
`.github/agents/` 直下に配置します。
outputsセクションの各エントリごとにファイルを生成し、nameを末尾に追加します。

例:
    .github_copilot_template/coder/script/.agent.md (outputs: [default, debug])
    → .github/agents/coder.script.default.agent.md
    → .github/agents/coder.script.debug.agent.md

==============================================================================
変数置換ルール
==============================================================================
${custom:name} 形式の変数を outputs の値で置換します。
- 値が "default" の場合: ${input:name:"custom_inputsのdefault値"} に変換
- それ以外の場合: その値で直接置換

==============================================================================
設定ファイル
==============================================================================
scripts/github_copilot/template_handle/config/deploy_agents.yaml.example

==============================================================================
使用方法
==============================================================================
    python scripts/github_copilot/template_handle/deploy_agents.py [OPTIONS] [CONFIG]

引数:
    CONFIG              デプロイ対象を指定するYAML設定ファイル（任意）

オプション:
    --overwrite         既存ファイルを上書き（デフォルト）
    --no-overwrite      既存ファイルをスキップ
    --clean             デプロイ前に .github/agents/ 内の全ファイルを削除

実行例:
    # 全エージェントをデプロイ（既存ファイルは上書き）
    python scripts/github_copilot/template_handle/deploy_agents.py

    # 設定ファイルで指定したエージェントのみデプロイ
    python scripts/github_copilot/template_handle/deploy_agents.py agents-config.yaml

    # クリーンデプロイ（削除後にデプロイ）
    python scripts/github_copilot/template_handle/deploy_agents.py --clean
"""

import sys
from pathlib import Path

from ..util.path_utils import get_project_root, path_to_dot_notation
from ..util.substitution_utils import (
    get_output_values_without_name,
    substitute_custom_variables,
)
from ..util.template_utils import (
    find_agent_files,
    get_agents_dir,
    get_template_base_dir,
)
from ..util.yaml_utils import (
    extract_custom_inputs,
    extract_outputs,
    parse_frontmatter,
    parse_yaml_include,
    rebuild_content_with_frontmatter,
    remove_custom_sections_from_frontmatter,
)


def filter_by_patterns(
    agent_files: list[Path], patterns: list[str], template_dir: Path
) -> list[Path]:
    """
    パターンに基づいてエージェントファイルをフィルタリングする

    パターン形式:
    - ディレクトリパターン（/で終わる）: そのディレクトリ配下のすべてのファイルを含む
      例: "coder/" は .github_copilot_template/coder/ 配下のすべてを含む
    - ファイルパターン: 特定のサブディレクトリを指定
      例: "general/basic" は .github_copilot_template/general/basic/.agent.md を含む

    Args:
        agent_files: フィルタリング対象のファイルリスト
        patterns: フィルタリングパターンのリスト
        template_dir: テンプレートディレクトリのパス

    Returns:
        list[Path]: フィルタリング後のファイルリスト
    """
    filtered: list[Path] = []

    for agent_file in agent_files:
        # テンプレートディレクトリからの相対パス（.agent.mdを除いた親ディレクトリ）
        relative_path = agent_file.parent.relative_to(template_dir)

        for pattern in patterns:
            if pattern.endswith("/"):
                # ディレクトリパターン: パターンで始まるパスをすべて含む
                pattern_dir = pattern.rstrip("/")
                if (
                    str(relative_path).startswith(pattern_dir)
                    or str(relative_path) == pattern_dir
                ):
                    filtered.append(agent_file)
                    break
            else:
                # ファイルパターン: 完全一致
                if str(relative_path) == pattern:
                    filtered.append(agent_file)
                    break

    return filtered


def generate_dest_filename(
    agent_file: Path, template_dir: Path, output_name: str
) -> str:
    """
    ソースファイルのパスから宛先ファイル名を生成する

    例: .github_copilot_template/coder/script/.agent.md (output_name: "default")
        -> coder.script.default.agent.md

    Args:
        agent_file: ソースの.agent.mdファイルパス
        template_dir: テンプレートディレクトリのパス
        output_name: outputエントリのname

    Returns:
        str: 宛先ファイル名
    """
    agent_name = path_to_dot_notation(agent_file.parent, template_dir)
    return f"{agent_name}.{output_name}.agent.md"


def process_agent_file(
    agent_file: Path,
    template_dir: Path,
    dest_dir: Path,
    overwrite: bool,
) -> tuple[list[str], list[str], list[str]]:
    """
    単一のエージェントファイルを処理する

    Args:
        agent_file: 処理対象の.agent.mdファイルパス
        template_dir: テンプレートディレクトリのパス
        dest_dir: 宛先ディレクトリのパス
        overwrite: 既存ファイルを上書きするか

    Returns:
        tuple: (コピー成功, スキップ, エラー) のリスト
    """
    copied: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    try:
        # ファイル内容を読み込む
        content = agent_file.read_text(encoding="utf-8")

        # フロントマターを解析
        try:
            frontmatter, body = parse_frontmatter(content)
        except ValueError as e:
            errors.append(f"{agent_file}: {e}")
            return copied, skipped, errors

        # outputsセクションを取得（なければエラー）
        try:
            outputs = extract_outputs(frontmatter)
        except ValueError as e:
            errors.append(f"{agent_file}: {e}")
            return copied, skipped, errors

        # custom_inputsを取得
        custom_inputs = extract_custom_inputs(frontmatter)

        # custom_inputsとoutputsを除いたフロントマターを作成
        cleaned_frontmatter = remove_custom_sections_from_frontmatter(frontmatter)

        # 各outputエントリごとにファイルを生成
        for output_entry in outputs:
            output_name = output_entry.get("name")
            if not output_name:
                errors.append(f"{agent_file}: outputエントリにnameがありません")
                continue

            # 宛先ファイル名を生成
            dest_filename = generate_dest_filename(
                agent_file, template_dir, output_name
            )
            dest_path = dest_dir / dest_filename

            # 既存チェック
            if dest_path.exists() and not overwrite:
                skipped.append(f"{agent_file} -> {dest_path} (既に存在)")
                continue

            # 変数置換用の値を取得（nameを除く）
            output_values = get_output_values_without_name(output_entry)

            # 本文の変数を置換
            substituted_body = substitute_custom_variables(
                body, output_values, custom_inputs
            )

            # 最終コンテンツを生成
            final_content = rebuild_content_with_frontmatter(
                cleaned_frontmatter, substituted_body
            )

            # ファイルを書き込む
            dest_path.write_text(final_content, encoding="utf-8")
            copied.append(f"{agent_file} -> {dest_path}")

    except Exception as e:
        errors.append(f"{agent_file}: {e}")

    return copied, skipped, errors


def deploy_agents(
    template_dir: Path,
    dest_dir: Path,
    patterns: list[str] | None = None,
    overwrite: bool = True,
    clean: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    エージェントファイルを展開する

    各.agent.mdファイルのoutputsセクションに基づいて、
    複数のファイルを生成します。

    Args:
        template_dir: テンプレートディレクトリのパス
        dest_dir: 宛先ディレクトリのパス
        patterns: フィルタリングパターン（Noneの場合は全て）
        overwrite: 既存ファイルを上書きするか
        clean: デプロイ前に宛先を削除するか

    Returns:
        tuple: (コピー成功, スキップ, エラー, 削除済み) のリスト
    """
    copied: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    deleted: list[str] = []

    # すべての.agent.mdファイルを取得
    agent_files = find_agent_files(template_dir)

    # パターンが指定されている場合はフィルタリング
    if patterns:
        agent_files = filter_by_patterns(agent_files, patterns, template_dir)

    # クリーンモード：宛先ディレクトリのすべてのファイルを削除
    if clean and dest_dir.exists():
        for file in dest_dir.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    deleted.append(str(file))
                except Exception as e:
                    errors.append(f"削除失敗 {file}: {e}")

    # 宛先ディレクトリを作成
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 各エージェントファイルを処理
    for agent_file in agent_files:
        file_copied, file_skipped, file_errors = process_agent_file(
            agent_file, template_dir, dest_dir, overwrite
        )
        copied.extend(file_copied)
        skipped.extend(file_skipped)
        errors.extend(file_errors)

    return copied, skipped, errors, deleted


def print_summary(
    copied: list[str],
    skipped: list[str],
    errors: list[str],
    deleted: list[str],
) -> None:
    """
    デプロイ結果のサマリーを表示する

    Args:
        copied: コピー成功したファイルのリスト
        skipped: スキップしたファイルのリスト
        errors: エラーが発生したファイルのリスト
        deleted: 削除したファイルのリスト
    """
    print(f"\n{'=' * 60}")
    print("デプロイ結果")
    print(f"{'=' * 60}")

    if deleted:
        print(f"\n削除 ({len(deleted)} ファイル):")
        for item in deleted:
            print(f"  🗑 {item}")

    if copied:
        print(f"\nコピー ({len(copied)} ファイル):")
        for item in copied:
            print(f"  ✓ {item}")

    if skipped:
        print(f"\nスキップ ({len(skipped)} ファイル):")
        for item in skipped:
            print(f"  ⊘ {item}")

    if errors:
        print(f"\nエラー ({len(errors)} ファイル):")
        for item in errors:
            print(f"  ✗ {item}")

    print(f"\n{'=' * 60}")
    print(
        f"合計: {len(deleted)} 削除, {len(copied)} コピー, "
        f"{len(skipped)} スキップ, {len(errors)} エラー"
    )


def main() -> None:
    """
    メイン処理: スクリプトのエントリーポイント
    """
    project_root = get_project_root()
    template_dir = get_template_base_dir(project_root)
    dest_dir = get_agents_dir(project_root)

    # デフォルト設定
    config_path: Path | None = None
    overwrite = True
    clean = False

    # 引数を解析
    args = sys.argv[1:]
    for arg in args:
        if arg == "--overwrite":
            overwrite = True
        elif arg == "--no-overwrite":
            overwrite = False
        elif arg == "--clean":
            clean = True
        elif arg.endswith(".yaml") or arg.endswith(".yml"):
            config_path = Path(arg)
            if not config_path.is_absolute():
                config_path = project_root / config_path

    # テンプレートディレクトリの存在確認
    if not template_dir.exists():
        print(f"エラー: テンプレートディレクトリが見つかりません: {template_dir}")
        sys.exit(1)

    # パターンを取得
    patterns: list[str] | None = None
    if config_path:
        if not config_path.exists():
            print(f"エラー: 設定ファイルが見つかりません: {config_path}")
            sys.exit(1)
        patterns = parse_yaml_include(config_path)
        print(f"設定ファイル: {config_path}")
        print(f"対象パターン: {patterns}")

    if clean:
        print("クリーンモード: 宛先ディレクトリの全ファイルを削除します")

    # エージェントを展開
    copied, skipped, errors, deleted = deploy_agents(
        template_dir, dest_dir, patterns, overwrite, clean
    )

    # 結果を表示
    print_summary(copied, skipped, errors, deleted)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
