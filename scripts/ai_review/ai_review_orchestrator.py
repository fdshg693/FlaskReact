"""
AI Review Orchestrator

Diff生成とAIレビューを統合し、エンドツーエンドの
コードレビュープロセスを実行する
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import ConfigurationError, ReviewConfig
from generate_ai_review import AIReviewer, AIReviewError
from generate_diff import DiffGenerationError, DiffGenerator

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PRReviewOrchestrator:
    """PR/コードレビューの全体フローを制御するクラス"""

    def __init__(self, config: ReviewConfig, verbose: bool = False):
        """
        オーケストレーターを初期化

        Args:
            config: ReviewConfig設定オブジェクト
            verbose: 詳細ログを表示するか
        """
        self.config = config
        self.verbose = verbose

        # ロガーレベルの設定
        if verbose:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("generate_diff").setLevel(logging.DEBUG)
            logging.getLogger("generate_ai_review").setLevel(logging.DEBUG)

        # 各モジュールの初期化
        self.diff_generator = DiffGenerator(config)
        self.ai_reviewer = AIReviewer(config)

    def run(
        self,
        base_branch: Optional[str] = None,
        model: Optional[str] = None,
        prompt_file: Optional[Path] = None,
        max_lines: Optional[int] = None,
        quiet: bool = False,
    ) -> bool:
        """
        AIレビューの全体フローを実行

        実行フロー:
        1. 環境変数の検証（初期化時に完了）
        2. Diff生成
        3. 変更有無チェック
        4. AIレビュー生成
        5. 結果表示

        Args:
            base_branch: ベースブランチ名
            model: 使用するAIモデル名
            prompt_file: カスタムプロンプトファイルのパス
            max_lines: 処理する最大行数
            quiet: 最小限の出力のみ

        Returns:
            bool: 成功時True、失敗時False
        """
        try:
            # ステップ1: 環境変数検証（既に初期化時に完了）
            if not quiet:
                logger.info("=" * 60)
                logger.info("AI Code Review Orchestrator")
                logger.info("=" * 60)

            # ステップ2: Diff生成
            if not quiet:
                logger.info("\n[1/3] Generating diff...")

            diff_result = self._generate_diff(base_branch)

            if not diff_result["has_changes"]:
                if not quiet:
                    logger.info("\n✓ No changes detected. Review skipped.")
                return True

            diff_path = diff_result["diff_path"]
            if not quiet:
                logger.info(f"✓ Diff generated: {diff_path}")
                logger.info(f"  Lines: {diff_result.get('line_count', 'N/A')}")
                logger.info(f"  Base branch: {diff_result['base_branch']}")

            # ステップ3: AIレビュー生成
            if not quiet:
                logger.info("\n[2/3] Generating AI review...")

            review_result = self._generate_review(
                diff_path,
                model=model,
                prompt_file=prompt_file,
                max_lines=max_lines,
                quiet=quiet,
            )

            # ステップ4: 結果表示
            if not quiet:
                logger.info("\n[3/3] Review complete!")
                logger.info("=" * 60)
                logger.info(f"✓ Review saved to: {review_result['output_path']}")
                if review_result.get("stats"):
                    stats = review_result["stats"]
                    logger.info(f"  Lines: {stats.get('lines', 'N/A')}")
                    logger.info(f"  Characters: {stats.get('chars', 'N/A')}")
                    logger.info(
                        f"  Estimated tokens: {stats.get('estimated_tokens', 'N/A')}"
                    )
                logger.info("=" * 60)

            return True

        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            return False
        except DiffGenerationError as e:
            logger.error(f"Diff generation failed: {e}")
            return False
        except AIReviewError as e:
            logger.error(f"AI review generation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if self.verbose:
                logger.exception("Detailed traceback:")
            return False

    def _generate_diff(self, base_branch: Optional[str] = None) -> dict:
        """
        Diff生成を実行

        Args:
            base_branch: ベースブランチ名

        Returns:
            dict: Diff生成結果
        """
        output_path = self.config.tmp_dir / "diff.patch"
        return self.diff_generator.generate_diff(
            base_branch=base_branch, output_path=output_path
        )

    def _generate_review(
        self,
        diff_path: Path,
        model: Optional[str] = None,
        prompt_file: Optional[Path] = None,
        max_lines: Optional[int] = None,
        quiet: bool = False,
    ) -> dict:
        """
        AIレビュー生成を実行

        Args:
            diff_path: Diffファイルのパス
            model: 使用するAIモデル名
            prompt_file: カスタムプロンプトファイルのパス
            max_lines: 処理する最大行数
            quiet: 詳細ログを抑制

        Returns:
            dict: レビュー生成結果
        """
        output_path = self.config.tmp_dir / "ai_review_output.md"

        # カスタムプロンプトの読み込み
        custom_prompt = None
        if prompt_file:
            if not prompt_file.exists():
                raise AIReviewError(f"Prompt file not found: {prompt_file}")
            custom_prompt = prompt_file.read_text(encoding="utf-8")

        # レビュー生成
        self.ai_reviewer.review_diff(
            diff_path=diff_path,
            output_path=output_path,
            custom_prompt=custom_prompt,
            model=model,
            max_lines=max_lines,
        )

        # 結果の統計情報を取得
        stats = None
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8")
            stats = {
                "lines": len(content.splitlines()),
                "chars": len(content),
                "estimated_tokens": len(content) // 4,  # 簡易推定
            }

        return {"output_path": output_path, "stats": stats}


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Generate AI-powered code review from git diff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review changes against main branch
  python ai-review_orchestrator.py

  # Review against specific branch
  python ai-review_orchestrator.py -b develop

  # Use custom model and prompt
  python ai-review_orchestrator.py --model gpt-4 --prompt-file custom_prompt.txt

  # Verbose output
  python ai-review_orchestrator.py -v

  # Quiet mode (minimal output)
  python ai-review_orchestrator.py -q
        """,
    )

    # 引数定義
    parser.add_argument(
        "base_branch",
        nargs="?",
        default=None,
        help="Base branch to compare against (default: auto-detect from main/master)",
    )
    parser.add_argument(
        "-b",
        "--base-branch",
        dest="base_branch_flag",
        help="Base branch to compare against (alternative syntax)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Minimal output (show only essential information)",
    )
    parser.add_argument(
        "--model", help="AI model to use for review (default: from config or gpt-4)"
    )
    parser.add_argument(
        "--prompt-file", type=Path, help="Path to custom prompt template file"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Maximum number of diff lines to process (default: unlimited)",
    )

    args = parser.parse_args()

    # verboseとquietの競合チェック
    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet cannot be used together")

    # base_branchの決定（位置引数とフラグの統合）
    base_branch = args.base_branch_flag or args.base_branch

    try:
        # 設定の初期化
        config = ReviewConfig()

        # オーケストレーターの初期化と実行
        orchestrator = PRReviewOrchestrator(config, verbose=args.verbose)
        success = orchestrator.run(
            base_branch=base_branch,
            model=args.model,
            prompt_file=args.prompt_file,
            max_lines=args.max_lines,
            quiet=args.quiet,
        )

        sys.exit(0 if success else 1)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.error(
            "Please check your .env file and ensure all required variables are set."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            logger.exception("Detailed traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
