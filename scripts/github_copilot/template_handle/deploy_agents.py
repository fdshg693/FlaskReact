"""
エージェントデプロイスクリプト
================================

概要:
    `.github_copilot_template/` 配下の `.agent.md` ファイルを
    `.github/agents/` へ展開するスクリプト。
    `.agent.md` 以外のファイルは無視される。

パス変換ルール:
    テンプレートディレクトリからの相対パスを `.` 区切りに変換して
    `.github/agents/` 直下に配置する。

    例:
        .github_copilot_template/coder/script/.agent.md
        → .github/agents/coder.script.agent.md

設定ファイル:
    scripts/github_copilot/template_handle/config/deploy_agents.yaml.example

使用方法:
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

    # 既存ファイルをスキップしてデプロイ
    python scripts/github_copilot/template_handle/deploy_agents.py --no-overwrite

    # クリーンデプロイ（削除後にデプロイ）
    python scripts/github_copilot/template_handle/deploy_agents.py --clean

    # 組み合わせ例
    python scripts/github_copilot/template_handle/deploy_agents.py config.yaml --clean
"""

from pathlib import Path
import shutil
import sys


def parse_yaml_include(yaml_path: Path) -> list[str]:
    """
    簡易YAMLパーサー：includeセクションのリストを取得する。
    標準ライブラリのみを使用するため、シンプルな形式のみ対応。
    """
    patterns = []
    in_include_section = False

    with yaml_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # コメントまたは空行はスキップ
            if not stripped or stripped.startswith("#"):
                continue

            # includeセクションの開始を検出
            if stripped.startswith("include:"):
                in_include_section = True
                continue

            # 別のセクションが始まったら終了
            if (
                not line.startswith(" ")
                and not line.startswith("\t")
                and ":" in stripped
            ):
                in_include_section = False
                continue

            # includeセクション内のリストアイテムを解析
            if in_include_section and stripped.startswith("-"):
                # コメント部分を除去
                item = stripped[1:].split("#")[0].strip()
                # クォートを除去
                item = item.strip("'\"")
                if item:
                    patterns.append(item)

    return patterns


def get_all_agent_files(template_dir: Path) -> list[Path]:
    """
    テンプレートディレクトリ配下のすべての.agent.mdファイルを取得する。
    """
    return list(template_dir.rglob(".agent.md"))


def filter_by_patterns(
    agent_files: list[Path], patterns: list[str], template_dir: Path
) -> list[Path]:
    """
    パターンに基づいてエージェントファイルをフィルタリングする。

    パターン形式:
    - ディレクトリパターン（/で終わる）: そのディレクトリ配下のすべてのファイルを含む
      例: "coder/" は .github_copilot_template/coder/ 配下のすべてを含む
    - ファイルパターン: 特定のサブディレクトリを指定
      例: "general/basic" は .github_copilot_template/general/basic/.agent.md を含む
    """
    filtered = []

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


def generate_dest_filename(agent_file: Path, template_dir: Path) -> str:
    """
    ソースファイルのパスから宛先ファイル名を生成する。

    例: .github_copilot_template/coder/script/.agent.md -> coder.script.agent.md
    """
    relative_path = agent_file.parent.relative_to(template_dir)
    # パスの区切りを.に変換
    name_parts = relative_path.parts
    return ".".join(name_parts) + ".agent.md"


def deploy_agents(
    template_dir: Path,
    dest_dir: Path,
    patterns: list[str] | None = None,
    overwrite: bool = True,
    clean: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    エージェントファイルを展開する。

    Returns:
        tuple[list[str], list[str], list[str], list[str]]: (コピー成功, スキップ, エラー, 削除済み)のリスト
    """
    copied = []
    skipped = []
    errors = []
    deleted = []

    # すべての.agent.mdファイルを取得
    agent_files = get_all_agent_files(template_dir)

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
                    errors.append(f"Failed to delete {file}: {e}")

    # 宛先ディレクトリを作成
    dest_dir.mkdir(parents=True, exist_ok=True)

    for agent_file in agent_files:
        dest_filename = generate_dest_filename(agent_file, template_dir)
        dest_path = dest_dir / dest_filename

        try:
            if dest_path.exists() and not overwrite:
                skipped.append(f"{agent_file} -> {dest_path} (already exists)")
                continue

            shutil.copy2(agent_file, dest_path)
            copied.append(f"{agent_file} -> {dest_path}")
        except Exception as e:
            errors.append(f"{agent_file}: {e}")

    return copied, skipped, errors, deleted


def main():
    # プロジェクトルートを特定（このスクリプトの3階層上）
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent

    template_dir = project_root / ".github_copilot_template"
    dest_dir = project_root / ".github" / "agents"

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
        print(f"Error: Template directory not found: {template_dir}")
        sys.exit(1)

    # パターンを取得
    patterns: list[str] | None = None
    if config_path:
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        patterns = parse_yaml_include(config_path)
        print(f"Using config: {config_path}")
        print(f"Include patterns: {patterns}")

    if clean:
        print("Clean mode enabled: will delete all existing files in destination")

    # エージェントを展開
    copied, skipped, errors, deleted = deploy_agents(
        template_dir, dest_dir, patterns, overwrite, clean
    )

    # 結果を表示
    print(f"\n{'=' * 60}")
    print("Deployment Summary")
    print(f"{'=' * 60}")

    if deleted:
        print(f"\nDeleted ({len(deleted)} files):")
        for item in deleted:
            print(f"  🗑 {item}")

    if copied:
        print(f"\nCopied ({len(copied)} files):")
        for item in copied:
            print(f"  ✓ {item}")

    if skipped:
        print(f"\nSkipped ({len(skipped)} files):")
        for item in skipped:
            print(f"  ⊘ {item}")

    if errors:
        print(f"\nErrors ({len(errors)} files):")
        for item in errors:
            print(f"  ✗ {item}")

    print(f"\n{'=' * 60}")
    print(
        f"Total: {len(deleted)} deleted, {len(copied)} copied, {len(skipped)} skipped, {len(errors)} errors"
    )

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
