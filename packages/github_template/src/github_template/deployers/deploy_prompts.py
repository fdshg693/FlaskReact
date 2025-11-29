"""
プロンプトデプロイスクリプト

==============================================================================
概要
==============================================================================
`.github_copilot_template/` 配下の `.prompt.md` ファイルを
`.github/prompts/` へ展開するスクリプトです。

==============================================================================
パス変換ルール
==============================================================================
テンプレートディレクトリからの相対パスを `.` 区切りに変換して
`.github/prompts/` 直下に配置します。

例:
    .github_copilot_template/coder/script/default.prompt.md
    → .github/prompts/coder.script.default.prompt.md

    .github_copilot_template/coder/script/refactor.prompt.md
    → .github/prompts/coder.script.refactor.prompt.md

==============================================================================
エージェント検証
==============================================================================
各プロンプトファイルのフロントマターに記載された `agent` フィールドを検証し、
対応するエージェントファイルが `.github/agents/` に存在するかチェックします。
存在しない場合は警告を出力してスキップします。

==============================================================================
使用方法
==============================================================================
    python scripts/github_copilot/template_handle/deploy_prompts.py [OPTIONS]

オプション:
    --overwrite         既存ファイルを上書き（デフォルト）
    --no-overwrite      既存ファイルをスキップ
    --clean             デプロイ前に .github/prompts/ 内の全ファイルを削除
    --skip-validation   エージェント検証をスキップ

実行例:
    # 全プロンプトをデプロイ（既存ファイルは上書き）
    python scripts/github_copilot/template_handle/deploy_prompts.py

    # クリーンデプロイ（削除後にデプロイ）
    python scripts/github_copilot/template_handle/deploy_prompts.py --clean

    # エージェント検証なしでデプロイ
    python scripts/github_copilot/template_handle/deploy_prompts.py --skip-validation
"""

import sys
from pathlib import Path

from ..util.path_utils import get_project_root
from ..util.template_utils import (
    find_prompt_files,
    get_agents_dir,
    get_prompts_dir,
    get_template_base_dir,
)
from ..util.yaml_utils import parse_frontmatter


def validate_agent_exists(agent_name: str, agents_dir: Path) -> bool:
    """
    指定されたエージェントが存在するか検証する（完全一致のみ）

    Args:
        agent_name: エージェント名（例: coder.script.default）
        agents_dir: エージェントディレクトリのパス

    Returns:
        bool: エージェントファイルが存在する場合True
    """
    agent_file = agents_dir / f"{agent_name}.agent.md"
    return agent_file.exists()


def generate_dest_filename(prompt_file: Path, template_dir: Path) -> str:
    """
    ソースファイルのパスから宛先ファイル名を生成する

    例: .github_copilot_template/coder/script/default.prompt.md
        -> coder.script.default.prompt.md

    Args:
        prompt_file: ソースの.prompt.mdファイルパス
        template_dir: テンプレートディレクトリのパス

    Returns:
        str: 宛先ファイル名
    """
    # プロンプトファイルの親ディレクトリのパスをドット記法に変換
    relative_dir = prompt_file.parent.relative_to(template_dir)
    dir_notation = str(relative_dir).replace("/", ".").replace("\\", ".")

    # ファイル名を取得
    filename = prompt_file.name

    # 結合して返す
    return f"{dir_notation}.{filename}"


def process_prompt_file(
    prompt_file: Path,
    template_dir: Path,
    dest_dir: Path,
    agents_dir: Path,
    overwrite: bool,
    skip_validation: bool,
) -> tuple[str | None, str | None, str | None]:
    """
    単一のプロンプトファイルを処理する

    Args:
        prompt_file: 処理対象の.prompt.mdファイルパス
        template_dir: テンプレートディレクトリのパス
        dest_dir: 宛先ディレクトリのパス
        agents_dir: エージェントディレクトリのパス
        overwrite: 既存ファイルを上書きするか
        skip_validation: エージェント検証をスキップするか

    Returns:
        tuple: (コピー成功, スキップ, エラー) のいずれか1つが文字列、他はNone
    """
    try:
        # ファイル内容を読み込む
        content = prompt_file.read_text(encoding="utf-8")

        # フロントマターを解析
        try:
            frontmatter, body = parse_frontmatter(content)
        except ValueError as e:
            return None, None, f"{prompt_file}: {e}"

        # agentフィールドを検証
        agent_name = frontmatter.get("agent")
        if not agent_name:
            return None, None, f"{prompt_file}: agentフィールドが見つかりません"

        # エージェントの存在確認（必須チェック、skip_validationでのみスキップ可能）
        if not skip_validation:
            if not validate_agent_exists(agent_name, agents_dir):
                return (
                    None,
                    None,
                    f"{prompt_file}: エージェント '{agent_name}' が存在しません",
                )

        # 宛先ファイル名を生成
        dest_filename = generate_dest_filename(prompt_file, template_dir)
        dest_path = dest_dir / dest_filename

        # 既存チェック
        if dest_path.exists() and not overwrite:
            return None, f"{prompt_file} -> {dest_path} (既に存在)", None

        # ファイルをコピー
        dest_path.write_text(content, encoding="utf-8")
        return f"{prompt_file} -> {dest_path}", None, None

    except Exception as e:
        return None, None, f"{prompt_file}: {e}"


def deploy_prompts(
    template_dir: Path,
    dest_dir: Path,
    agents_dir: Path,
    overwrite: bool = True,
    clean: bool = False,
    skip_validation: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    プロンプトファイルを展開する

    Args:
        template_dir: テンプレートディレクトリのパス
        dest_dir: 宛先ディレクトリのパス
        agents_dir: エージェントディレクトリのパス
        overwrite: 既存ファイルを上書きするか
        clean: デプロイ前に宛先ディレクトリをクリーンアップするか
        skip_validation: エージェント検証をスキップするか

    Returns:
        tuple: (コピー成功リスト, スキップリスト, エラーリスト)
    """
    copied: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    # クリーンアップ
    if clean and dest_dir.exists():
        print(f"クリーンアップ: {dest_dir}")
        for file in dest_dir.glob("*.prompt.md"):
            file.unlink()
            print(f"  削除: {file}")

    # 宛先ディレクトリを作成
    dest_dir.mkdir(parents=True, exist_ok=True)

    # プロンプトファイルを検索
    prompt_files = find_prompt_files(template_dir)

    if not prompt_files:
        print(f"警告: プロンプトファイルが見つかりませんでした: {template_dir}")
        return copied, skipped, errors

    # 各ファイルを処理
    for prompt_file in prompt_files:
        result_copied, result_skipped, result_error = process_prompt_file(
            prompt_file,
            template_dir,
            dest_dir,
            agents_dir,
            overwrite,
            skip_validation,
        )

        if result_copied:
            copied.append(result_copied)
        if result_skipped:
            skipped.append(result_skipped)
        if result_error:
            errors.append(result_error)

    return copied, skipped, errors


def main() -> None:
    """
    メイン関数
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="プロンプトファイルを.github/prompts/にデプロイする"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="既存ファイルを上書き（デフォルト）",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="既存ファイルをスキップ",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="デプロイ前に宛先ディレクトリをクリーンアップ",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="エージェント検証をスキップ"
    )

    args = parser.parse_args()

    # プロジェクトルートを取得
    project_root = get_project_root()

    # ディレクトリパスを取得
    template_dir = get_template_base_dir(project_root)
    dest_dir = get_prompts_dir(project_root)
    agents_dir = get_agents_dir(project_root)

    print("=" * 60)
    print("プロンプトファイルデプロイ")
    print("=" * 60)
    print(f"テンプレートディレクトリ: {template_dir}")
    print(f"宛先ディレクトリ: {dest_dir}")
    print(f"エージェントディレクトリ: {agents_dir}")
    print(f"上書きモード: {args.overwrite}")
    print(f"クリーンモード: {args.clean}")
    print(f"エージェント検証: {not args.skip_validation}")
    print("=" * 60)

    # デプロイ実行
    copied, skipped, errors = deploy_prompts(
        template_dir,
        dest_dir,
        agents_dir,
        overwrite=args.overwrite,
        clean=args.clean,
        skip_validation=args.skip_validation,
    )

    # 結果を表示
    print("\n【コピー成功】")
    if copied:
        for item in copied:
            print(f"  ✓ {item}")
    else:
        print("  (なし)")

    if skipped:
        print("\n【スキップ】")
        for item in skipped:
            print(f"  - {item}")

    if errors:
        print("\n【エラー】")
        for item in errors:
            print(f"  ✗ {item}")

    print("\n" + "=" * 60)
    print(f"完了: {len(copied)} 個のファイルをデプロイしました")
    if skipped:
        print(f"スキップ: {len(skipped)} 個")
    if errors:
        print(f"エラー: {len(errors)} 個")
        sys.exit(1)


if __name__ == "__main__":
    main()
