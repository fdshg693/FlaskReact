"""
Diff生成モジュール

GitPythonを使用してブランチ間のdiffを生成する
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from git import Repo, GitCommandError, InvalidGitRepositoryError
from config import ReviewConfig, ConfigurationError


# ロガー設定
logger = logging.getLogger(__name__)


class DiffGenerationError(Exception):
    """Diff生成に関するエラー"""
    pass


class DiffGenerator:
    """Git差分を生成するクラス"""

    def __init__(self, config: ReviewConfig):
        """
        DiffGeneratorを初期化

        Args:
            config: ReviewConfig設定オブジェクト

        Raises:
            InvalidGitRepositoryError: Gitリポジトリでない場合
        """
        self.config = config
        try:
            self.repo = Repo(config.project_root, search_parent_directories=True)
        except InvalidGitRepositoryError as e:
            raise DiffGenerationError(
                f"Gitリポジトリが見つかりません: {config.project_root}\n"
                f"プロジェクトルートがGitリポジトリであることを確認してください。\n"
                f"'git init' または 'git clone' でリポジトリを初期化してください。"
            ) from e

    def _determine_base_branch(self, base_branch: Optional[str] = None) -> str:
        """
        ベースブランチを決定する

        Args:
            base_branch: 明示的に指定されたベースブランチ

        Returns:
            決定されたベースブランチ名

        優先順位:
        1. 引数で指定されたbase_branch
        2. 環境変数 PR_BASE_REF
        3. 環境変数 INPUT_TARGET
        4. デフォルト: "main"
        """
        if base_branch and base_branch != "null":
            return base_branch

        pr_base_ref = os.getenv("PR_BASE_REF", "")
        if pr_base_ref and pr_base_ref != "null":
            return pr_base_ref

        input_target = os.getenv("INPUT_TARGET", "")
        if input_target and input_target != "null":
            return input_target

        return "main"

    def _fetch_base_branch(self, base_branch: str) -> None:
        """
        ベースブランチをリモートからフェッチする

        Args:
            base_branch: フェッチするブランチ名

        Raises:
            DiffGenerationError: フェッチに失敗した場合
        """
        try:
            logger.info(f"Fetching base branch: {base_branch}")
            origin = self.repo.remote("origin")
            origin.fetch(base_branch)
        except GitCommandError as e:
            # 利用可能なリモートブランチをリスト
            try:
                remote_branches = [ref.name for ref in self.repo.remote().refs]
                branches_info = "\n  ".join(remote_branches)
                error_msg = (
                    f"ベースブランチ '{base_branch}' のフェッチに失敗しました。\n"
                    f"ブランチ名が正しいか確認してください。\n"
                    f"利用可能なリモートブランチ:\n  {branches_info}"
                )
            except Exception:
                error_msg = (
                    f"ベースブランチ '{base_branch}' のフェッチに失敗しました。\n"
                    f"ブランチが存在するか、リモート接続が正しいか確認してください。"
                )

            raise DiffGenerationError(error_msg) from e

    def _generate_diff_content(self, base_branch: str) -> str:
        """
        diffの内容を生成する

        Args:
            base_branch: 比較対象のベースブランチ

        Returns:
            diff文字列

        Raises:
            DiffGenerationError: diff生成に失敗した場合
        """
        try:
            # git diff origin/base_branch...HEAD と同等
            diff_content = self.repo.git.diff(
                f"origin/{base_branch}...HEAD",
                unified=3,
                no_color=True,
                ignore_space_change=True,
            )
            return diff_content
        except GitCommandError as e:
            raise DiffGenerationError(
                f"diff生成に失敗しました: {e}"
            ) from e

    def generate_diff(
        self,
        base_branch: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Gitのdiffを生成してファイルに保存する

        Args:
            base_branch: 比較対象のベースブランチ（省略時は自動判定）
            output_path: 出力先パス（省略時は config.get_diff_path()）

        Returns:
            結果を含む辞書:
            {
                "has_changes": bool,        # 変更があるか
                "diff_path": Path,          # 出力先パス
                "line_count": int,          # 差分の行数
                "base_branch": str,         # 使用したベースブランチ
            }

        Raises:
            DiffGenerationError: diff生成に失敗した場合
        """
        # ベースブランチの決定
        resolved_base_branch = self._determine_base_branch(base_branch)
        logger.info(f"Comparing against base branch: {resolved_base_branch}")

        # ベースブランチのフェッチ
        self._fetch_base_branch(resolved_base_branch)

        # diff生成
        diff_content = self._generate_diff_content(resolved_base_branch)

        # 出力先パスの決定
        if output_path is None:
            output_path = self.config.get_diff_path()

        # 出力ディレクトリの作成
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ファイルに書き込み
        output_path.write_text(diff_content, encoding="utf-8")

        # 差分の有無を確認
        has_changes = len(diff_content.strip()) > 0
        line_count = len(diff_content.splitlines()) if has_changes else 0

        return {
            "has_changes": has_changes,
            "diff_path": output_path,
            "line_count": line_count,
            "base_branch": resolved_base_branch,
        }


def setup_logging(verbose: bool = False) -> None:
    """ロギングの設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """メイン関数（CLI実行時）"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Git diffを生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--base-branch",
        help="比較対象のベースブランチ（デフォルト: main）",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="出力先ファイルパス（デフォルト、tmp/diff.patch）",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="詳細ログを表示",
        action="store_true",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.verbose)

    try:
        # 設定の読み込み
        config = ReviewConfig()
        logger.debug(f"設定読み込み完了:\n{config}")

        # Diff生成
        generator = DiffGenerator(config)
        result = generator.generate_diff(
            base_branch=args.base_branch,
            output_path=args.output,
        )

        # 結果の表示
        if result["has_changes"]:
            logger.info(
                f"✅ diff生成完了 ({result['line_count']} 行)\n"
                f"   出力先: {result['diff_path']}\n"
                f"   ベースブランチ: {result['base_branch']}"
            )
            return 0
        else:
            logger.info(
                f"ℹ️ 変更なし（ベースブランチ: {result['base_branch']}）"
            )
            return 0

    except ConfigurationError as e:
        logger.error(f"❌ 設定エラー: {e}")
        return 1
    except DiffGenerationError as e:
        logger.error(f"❌ Diff生成エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\n⚠️ 中断されました")
        return 130
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
