"""
AI Review Generator Module

OpenAI APIを使用してコードdiffのレビューを生成する
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import APITimeoutError, OpenAI, OpenAIError, RateLimitError

from config import ConfigurationError, ReviewConfig

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# デフォルトのレビュープロンプトテンプレート
DEFAULT_REVIEW_PROMPT = """You are an experienced software engineer. Please review the following code diff in detail and analyze it from the following perspectives in English:

1. Code Quality: Readability, maintainability, performance
2. Security: Potential vulnerabilities and security risks
3. Best Practices: Language and framework recommendations
4. Bug Potential: Logic errors and exception handling issues
5. Improvement Suggestions: Specific improvement proposals and refactoring suggestions

Output Format:
- Point out issues specifically and include relevant line numbers
- Provide implementable concrete examples for improvement suggestions
- Clearly indicate importance level (High, Medium, Low)

Code Diff:
"""


class AIReviewError(Exception):
    """AIレビュー生成関連のエラー"""

    pass


class AIReviewer:
    """OpenAI APIを使用してコードレビューを生成するクラス"""

    # トークン推定の定数（おおよそ1トークン = 4文字）
    CHARS_PER_TOKEN = 4
    # 安全なトークン制限（レスポンス用に余裕を持たせる）
    SAFE_TOKEN_LIMIT = 25000

    def __init__(
        self, config: ReviewConfig, max_retries: int = 3, retry_delay: int = 5
    ):
        """
        AIReviewerを初期化する

        Args:
            config: ReviewConfig インスタンス
            max_retries: API呼び出しの最大リトライ回数
            retry_delay: リトライ間の待機時間（秒）
        """
        self.config = config
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # OpenAI クライアントの初期化
        self.client = OpenAI(api_key=config.openai_api_key)

    def estimate_tokens(self, text: str) -> int:
        """
        テキストのトークン数を推定する

        Args:
            text: トークン数を推定するテキスト

        Returns:
            推定トークン数
        """
        return len(text) // self.CHARS_PER_TOKEN

    def create_prompt(
        self, diff_content: str, custom_prompt: Optional[str] = None
    ) -> str:
        """
        レビュープロンプトを生成する

        Args:
            diff_content: diffの内容
            custom_prompt: カスタムプロンプト（省略時はデフォルト）

        Returns:
            完成したプロンプト文字列
        """
        base_prompt = custom_prompt if custom_prompt else DEFAULT_REVIEW_PROMPT
        return f"{base_prompt}\n\n{diff_content}"

    def _call_openai_api(self, prompt: str, model: Optional[str] = None) -> str:
        """
        OpenAI APIを呼び出してレビューを生成する

        Args:
            prompt: レビュー生成用のプロンプト
            model: 使用するAIモデル（省略時は config から取得）

        Returns:
            レビュー内容（Markdown形式）

        Raises:
            AIReviewError: API呼び出しに失敗した場合
        """
        # モデルの決定
        active_model = model or self.config.ai_model

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"🔄 OpenAI APIにリクエストを送信中 "
                    f"(model: {active_model}, attempt: {attempt}/{self.max_retries})..."
                )

                response = self.client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful and constructive code reviewer. Please provide detailed and practical feedback.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                # レスポンスからコンテンツを抽出
                if not response.choices:
                    raise AIReviewError("APIレスポンスにコンテンツが含まれていません")

                content = response.choices[0].message.content

                if not content or content.strip() == "":
                    raise AIReviewError("APIから空のレビューが返されました")

                logger.info("✅ レビューの生成に成功しました")
                return content

            except RateLimitError as e:
                logger.warning(f"⚠️ レート制限エラー: {e}")
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * attempt
                    logger.info(f"⏳ {wait_time}秒待機してリトライします...")
                    time.sleep(wait_time)
                else:
                    raise AIReviewError(
                        f"レート制限エラー: 最大リトライ回数に達しました\n{e}"
                    ) from e

            except APITimeoutError as e:
                logger.warning(f"⚠️ タイムアウトエラー: {e}")
                if attempt < self.max_retries:
                    logger.info(f"⏳ {self.retry_delay}秒待機してリトライします...")
                    time.sleep(self.retry_delay)
                else:
                    raise AIReviewError(
                        f"タイムアウトエラー: 最大リトライ回数に達しました\n{e}"
                    ) from e

            except OpenAIError as e:
                logger.error(f"❌ OpenAI APIエラー: {e}")
                if attempt < self.max_retries:
                    logger.info(f"⏳ {self.retry_delay}秒待機してリトライします...")
                    time.sleep(self.retry_delay)
                else:
                    raise AIReviewError(f"OpenAI APIエラー: {e}") from e

            except Exception as e:
                raise AIReviewError(f"予期しないエラーが発生しました: {e}") from e

        raise AIReviewError("最大リトライ回数に達しました")

    def review_diff(
        self,
        diff_path: Path,
        output_path: Optional[Path] = None,
        custom_prompt: Optional[str] = None,
        max_lines: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        diffファイルを読み込んでAIレビューを生成する

        Args:
            diff_path: diffファイルのパス
            output_path: 出力ファイルのパス（省略時は config から取得）
            custom_prompt: カスタムプロンプト（省略時はデフォルト）
            max_lines: diffの最大行数制限（省略時は制限なし）
            model: 使用するAIモデル（省略時は config から取得）

        Returns:
            レビュー内容（Markdown形式）

        Raises:
            AIReviewError: レビュー生成に失敗した場合
        """
        # diffファイルの存在確認
        if not diff_path.exists():
            raise AIReviewError(f"Diffファイルが見つかりません: {diff_path}")

        # diffファイルの読み込み
        try:
            with open(diff_path, encoding="utf-8") as f:
                diff_content = f.read()
        except Exception as e:
            raise AIReviewError(f"Diffファイルの読み込みに失敗しました: {e}") from e

        # diffが空でないか確認
        if not diff_content.strip():
            raise AIReviewError("Diffファイルが空です")

        # 行数制限がある場合は切り詰める
        if max_lines:
            lines = diff_content.split("\n")
            if len(lines) > max_lines:
                logger.warning(
                    f"⚠️ Diffが大きすぎます。最初の{max_lines}行のみレビューします"
                )
                diff_content = "\n".join(lines[:max_lines])
                diff_content += (
                    f"\n\n... (残り {len(lines) - max_lines} 行は省略されました)"
                )

        # プロンプト生成
        prompt = self.create_prompt(diff_content, custom_prompt)

        # トークン数の推定と警告
        estimated_tokens = self.estimate_tokens(prompt)
        logger.info(f"📊 推定トークン数: {estimated_tokens:,}")

        if estimated_tokens > self.SAFE_TOKEN_LIMIT:
            logger.warning(
                f"⚠️ 推定トークン数（{estimated_tokens:,}）が安全な制限"
                f"（{self.SAFE_TOKEN_LIMIT:,}）を超えています"
            )
            logger.warning("💡 --max-lines オプションでdiffを制限することを推奨します")
            # 自動的に切り詰める提案
            suggested_lines = int(
                len(diff_content.split("\n")) * self.SAFE_TOKEN_LIMIT / estimated_tokens
            )
            logger.warning(f"💡 推奨: --max-lines {suggested_lines} を試してください")

        # APIを呼び出してレビュー生成
        review_content = self._call_openai_api(prompt, model=model)

        # 出力パスの決定
        if output_path is None:
            output_path = self.config.get_review_output_path()

        # レビューをファイルに保存
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(review_content)
            logger.info(f"📝 レビューを保存しました: {output_path}")
        except Exception as e:
            raise AIReviewError(f"レビューファイルの保存に失敗しました: {e}") from e

        return review_content

    def get_review_stats(self, review_content: str) -> Dict[str, Any]:
        """
        レビュー内容の統計情報を取得する

        Args:
            review_content: レビュー内容

        Returns:
            統計情報の辞書
        """
        lines = review_content.split("\n")
        return {
            "total_lines": len(lines),
            "total_chars": len(review_content),
            "total_words": len(review_content.split()),
        }


def main():
    """メイン関数（CLI実行用）"""
    parser = argparse.ArgumentParser(
        description="OpenAI APIを使用してコードdiffのレビューを生成します"
    )
    parser.add_argument(
        "diff_file",
        type=str,
        nargs="?",
        help="レビュー対象のdiffファイルのパス（デフォルト: tmp/diff.patch）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="レビュー出力ファイルのパス（デフォルト: tmp/ai_review_output.md）",
    )
    parser.add_argument(
        "-p", "--prompt-file", type=str, help="カスタムプロンプトファイルのパス"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細なログを表示")
    parser.add_argument(
        "--model", type=str, help="使用するAIモデル（環境変数AI_MODELを上書き）"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="API呼び出しの最大リトライ回数（デフォルト: 3）",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="リトライ間の待機時間（秒、デフォルト: 5）",
    )
    parser.add_argument(
        "--max-lines", type=int, help="diffの最大行数制限（大きなdiffを切り詰める）"
    )

    args = parser.parse_args()

    # ロギングレベルの設定
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("詳細ログモードが有効です")

    try:
        # 設定の読み込み
        logger.info("📋 設定を読み込んでいます...")
        config = ReviewConfig()

        # モデルの上書き
        if args.model:
            config.ai_model = args.model
            logger.info(f"モデルを上書きしました: {args.model}")

        # diffファイルパスの決定
        if args.diff_file:
            diff_path = Path(args.diff_file)
        else:
            diff_path = config.get_diff_path()
            logger.info(f"デフォルトのdiffファイルを使用します: {diff_path}")

        # 出力ファイルパスの決定
        output_path = Path(args.output) if args.output else None

        # カスタムプロンプトの読み込み
        custom_prompt = None
        if args.prompt_file:
            prompt_file = Path(args.prompt_file)
            if not prompt_file.exists():
                logger.error(f"❌ プロンプトファイルが見つかりません: {prompt_file}")
                return 1
            try:
                with open(prompt_file, encoding="utf-8") as f:
                    custom_prompt = f.read()
                logger.info(f"カスタムプロンプトを読み込みました: {prompt_file}")
            except Exception as e:
                logger.error(f"❌ プロンプトファイルの読み込みに失敗しました: {e}")
                return 1

        # AIレビューの生成
        reviewer = AIReviewer(
            config, max_retries=args.max_retries, retry_delay=args.retry_delay
        )
        review_content = reviewer.review_diff(
            diff_path, output_path, custom_prompt, max_lines=args.max_lines
        )

        # 統計情報の表示
        if args.verbose:
            stats = reviewer.get_review_stats(review_content)
            logger.info("📊 統計情報:")
            logger.info(f"  - 行数: {stats['total_lines']}")
            logger.info(f"  - 文字数: {stats['total_chars']}")
            logger.info(f"  - 単語数: {stats['total_words']}")

        logger.info("✅ AIレビューの生成が完了しました")
        return 0

    except ConfigurationError as e:
        logger.error(f"❌ 設定エラー: {e}")
        return 1
    except AIReviewError as e:
        logger.error(f"❌ レビュー生成エラー: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\n⚠️ ユーザーによって中断されました")
        return 130
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
