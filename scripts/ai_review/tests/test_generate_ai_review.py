"""
generate_ai_review.py の単体テスト

AIReviewerクラスのメソッドをテストする
"""

from unittest.mock import MagicMock, patch

import pytest
from openai import RateLimitError

from config import ReviewConfig
from generate_ai_review import DEFAULT_REVIEW_PROMPT, AIReviewer, AIReviewError


class TestAIReviewer:
    """AIReviewerクラスのテスト"""

    @pytest.fixture
    def config(self, mock_env, temp_dir, monkeypatch):
        """テスト用のReviewConfigを作成"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        return ReviewConfig()

    @pytest.fixture
    def mock_openai_client(self):
        """モックのOpenAIクライアントを作成"""
        mock_client = MagicMock()

        # 正常なレスポンスのモック
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a test review"
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    def test_init(self, config):
        """AIReviewerが正しく初期化されることを確認"""
        with patch("generate_ai_review.OpenAI") as MockOpenAI:
            reviewer = AIReviewer(config)

            assert reviewer.config == config
            assert reviewer.max_retries == 3
            assert reviewer.retry_delay == 5
            MockOpenAI.assert_called_once_with(api_key=config.openai_api_key)

    def test_init_with_custom_retries(self, config):
        """カスタムリトライ設定で初期化できることを確認"""
        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config, max_retries=5, retry_delay=10)

            assert reviewer.max_retries == 5
            assert reviewer.retry_delay == 10

    def test_estimate_tokens(self, config):
        """トークン数の推定が正しく動作することを確認"""
        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            text = "a" * 100
            estimated = reviewer.estimate_tokens(text)

            # 100文字 / 4 = 25トークン
            assert estimated == 25

    def test_create_prompt_default(self, config, sample_diff_content):
        """デフォルトプロンプトが正しく生成されることを確認"""
        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            prompt = reviewer.create_prompt(sample_diff_content)

            assert DEFAULT_REVIEW_PROMPT in prompt
            assert sample_diff_content in prompt

    def test_create_prompt_custom(self, config, sample_diff_content):
        """カスタムプロンプトが正しく使用されることを確認"""
        custom_prompt = "Custom review instructions"

        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            prompt = reviewer.create_prompt(sample_diff_content, custom_prompt)

            assert custom_prompt in prompt
            assert sample_diff_content in prompt
            assert DEFAULT_REVIEW_PROMPT not in prompt

    def test_call_openai_api_success(self, config, mock_openai_client):
        """OpenAI API呼び出しが成功することを確認"""
        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            result = reviewer._call_openai_api("Test prompt")

            assert result == "This is a test review"
            mock_openai_client.chat.completions.create.assert_called_once()

    def test_call_openai_api_with_custom_model(self, config, mock_openai_client):
        """カスタムモデル指定でAPI呼び出しが成功することを確認"""
        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            reviewer._call_openai_api("Test prompt", model="gpt-3.5-turbo")

            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-3.5-turbo"

    def test_call_openai_api_empty_response(self, config):
        """空のレスポンスが返された場合にエラーが発生することを確認"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        with patch("generate_ai_review.OpenAI", return_value=mock_client):
            reviewer = AIReviewer(config)

            with pytest.raises(AIReviewError, match="コンテンツが含まれていません"):
                reviewer._call_openai_api("Test prompt")

    def test_call_openai_api_rate_limit_retry(self, config):
        """レート制限エラー時にリトライすることを確認"""
        mock_client = MagicMock()

        # 最初はレート制限エラー、2回目は成功
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Success after retry"
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=MagicMock(), body=None),
            mock_response,
        ]

        with patch("generate_ai_review.OpenAI", return_value=mock_client):
            with patch("time.sleep"):  # sleepをモック
                reviewer = AIReviewer(config, retry_delay=0)

                result = reviewer._call_openai_api("Test prompt")

                assert result == "Success after retry"
                assert mock_client.chat.completions.create.call_count == 2

    def test_call_openai_api_rate_limit_max_retries(self, config):
        """最大リトライ回数に達した場合にエラーが発生することを確認"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=MagicMock(), body=None
        )

        with patch("generate_ai_review.OpenAI", return_value=mock_client):
            with patch("time.sleep"):
                reviewer = AIReviewer(config, max_retries=2, retry_delay=0)

                with pytest.raises(AIReviewError, match="最大リトライ回数に達しました"):
                    reviewer._call_openai_api("Test prompt")

    def test_review_diff_success(
        self, config, mock_openai_client, sample_diff_content, temp_dir
    ):
        """diff レビューが成功することを確認"""
        diff_path = temp_dir / "test.patch"
        diff_path.write_text(sample_diff_content, encoding="utf-8")
        output_path = temp_dir / "review.md"

        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            result = reviewer.review_diff(diff_path, output_path)

            assert result == "This is a test review"
            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == "This is a test review"

    def test_review_diff_file_not_found(self, config, temp_dir):
        """diffファイルが存在しない場合にエラーが発生することを確認"""
        diff_path = temp_dir / "nonexistent.patch"

        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            with pytest.raises(AIReviewError, match="見つかりません"):
                reviewer.review_diff(diff_path)

    def test_review_diff_empty_file(self, config, temp_dir):
        """空のdiffファイルの場合にエラーが発生することを確認"""
        diff_path = temp_dir / "empty.patch"
        diff_path.write_text("", encoding="utf-8")

        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            with pytest.raises(AIReviewError, match="空です"):
                reviewer.review_diff(diff_path)

    def test_review_diff_with_max_lines(self, config, mock_openai_client, temp_dir):
        """max_linesでdiffが切り詰められることを確認"""
        # 10行のdiffを作成
        diff_content = "\n".join([f"line {i}" for i in range(10)])
        diff_path = temp_dir / "test.patch"
        diff_path.write_text(diff_content, encoding="utf-8")

        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            reviewer.review_diff(diff_path, max_lines=5)

            # プロンプトが呼ばれたことを確認
            call_args = mock_openai_client.chat.completions.create.call_args
            prompt_content = call_args[1]["messages"][1]["content"]

            # 省略メッセージが含まれていることを確認
            assert "省略されました" in prompt_content

    def test_review_diff_default_output_path(
        self, config, mock_openai_client, sample_diff_content
    ):
        """デフォルトの出力パスが使用されることを確認"""
        diff_path = config.tmp_dir / "test.patch"
        diff_path.write_text(sample_diff_content, encoding="utf-8")

        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            reviewer.review_diff(diff_path)

            expected_output = config.get_review_output_path()
            assert expected_output.exists()

    def test_review_diff_with_custom_prompt(
        self, config, mock_openai_client, sample_diff_content, temp_dir
    ):
        """カスタムプロンプトが使用されることを確認"""
        diff_path = temp_dir / "test.patch"
        diff_path.write_text(sample_diff_content, encoding="utf-8")
        custom_prompt = "Custom review instructions"

        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            reviewer.review_diff(diff_path, custom_prompt=custom_prompt)

            call_args = mock_openai_client.chat.completions.create.call_args
            prompt_content = call_args[1]["messages"][1]["content"]

            assert custom_prompt in prompt_content

    def test_get_review_stats(self, config, sample_review_content):
        """レビュー統計が正しく取得できることを確認"""
        with patch("generate_ai_review.OpenAI"):
            reviewer = AIReviewer(config)

            stats = reviewer.get_review_stats(sample_review_content)

            assert "total_lines" in stats
            assert "total_chars" in stats
            assert "total_words" in stats
            assert stats["total_lines"] > 0
            assert stats["total_chars"] == len(sample_review_content)

    def test_token_warning_for_large_diff(self, config, mock_openai_client, temp_dir):
        """大きなdiffの場合に警告が表示されることを確認"""
        # 大きなdiffを生成（SAFE_TOKEN_LIMIT を超えるサイズ）
        large_diff = "x" * (
            AIReviewer.SAFE_TOKEN_LIMIT * AIReviewer.CHARS_PER_TOKEN + 1000
        )
        diff_path = temp_dir / "large.patch"
        diff_path.write_text(large_diff, encoding="utf-8")

        with patch("generate_ai_review.OpenAI", return_value=mock_openai_client):
            reviewer = AIReviewer(config)

            # ログ出力をキャプチャして警告が出ることを確認
            with patch("generate_ai_review.logger") as mock_logger:
                reviewer.review_diff(diff_path)

                # warning が呼ばれたことを確認
                assert any(
                    "推定トークン数" in str(call)
                    for call in mock_logger.warning.call_args_list
                )
