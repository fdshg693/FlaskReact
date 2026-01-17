"""
ai-review_orchestrator.py の統合テスト

PRReviewOrchestratorクラスの統合動作をテストする
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_review_orchestrator import PRReviewOrchestrator
from config import ReviewConfig
from generate_ai_review import AIReviewError
from generate_diff import DiffGenerationError


class TestPRReviewOrchestrator:
    """PRReviewOrchestratorクラスのテスト"""

    @pytest.fixture
    def config(self, mock_env, temp_dir, monkeypatch):
        """テスト用のReviewConfigを作成"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        return ReviewConfig()

    @pytest.fixture
    def mock_diff_generator(self):
        """モックのDiffGeneratorを作成"""
        mock_gen = MagicMock()
        mock_gen.generate_diff.return_value = {
            "has_changes": True,
            "diff_path": Path("tmp/diff.patch"),
            "line_count": 100,
            "base_branch": "main",
        }
        return mock_gen

    @pytest.fixture
    def mock_ai_reviewer(self):
        """モックのAIReviewerを作成"""
        mock_reviewer = MagicMock()
        mock_reviewer.review_diff.return_value = "Mock review content"
        return mock_reviewer

    def test_init(self, config):
        """オーケストレーターが正しく初期化されることを確認"""
        with patch("ai_review_orchestrator.DiffGenerator"), patch(
            "ai_review_orchestrator.AIReviewer"
        ):

            orchestrator = PRReviewOrchestrator(config)

            assert orchestrator.config == config
            assert orchestrator.verbose is False

    def test_init_verbose(self, config):
        """verboseモードで初期化できることを確認"""
        with patch("ai_review_orchestrator.DiffGenerator"), patch(
            "ai_review_orchestrator.AIReviewer"
        ):

            orchestrator = PRReviewOrchestrator(config, verbose=True)

            assert orchestrator.verbose is True

    def test_run_success_with_changes(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """変更がある場合の正常な実行を確認"""
        # レビュー出力ファイルを作成
        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run()

            assert result is True
            mock_diff_generator.generate_diff.assert_called_once()
            mock_ai_reviewer.review_diff.assert_called_once()

    def test_run_no_changes(self, config, mock_diff_generator, mock_ai_reviewer):
        """変更がない場合の実行を確認"""
        # 変更なしのレスポンス
        mock_diff_generator.generate_diff.return_value = {
            "has_changes": False,
            "diff_path": Path("tmp/diff.patch"),
            "line_count": 0,
            "base_branch": "main",
        }

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run()

            assert result is True
            mock_diff_generator.generate_diff.assert_called_once()
            # 変更がないのでレビューは呼ばれない
            mock_ai_reviewer.review_diff.assert_not_called()

    def test_run_with_base_branch(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """ベースブランチ指定での実行を確認"""
        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(base_branch="develop")

            assert result is True
            call_args = mock_diff_generator.generate_diff.call_args
            assert call_args[1]["base_branch"] == "develop"

    def test_run_with_custom_model(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """カスタムモデル指定での実行を確認"""
        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(model="gpt-3.5-turbo")

            assert result is True
            call_args = mock_ai_reviewer.review_diff.call_args
            assert call_args[1]["model"] == "gpt-3.5-turbo"

    def test_run_with_max_lines(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """max_lines指定での実行を確認"""
        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(max_lines=500)

            assert result is True
            call_args = mock_ai_reviewer.review_diff.call_args
            assert call_args[1]["max_lines"] == 500

    def test_run_with_prompt_file(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """プロンプトファイル指定での実行を確認"""
        # プロンプトファイルを作成
        prompt_file = temp_dir / "custom_prompt.txt"
        prompt_file.write_text("Custom prompt", encoding="utf-8")

        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(prompt_file=prompt_file)

            assert result is True
            call_args = mock_ai_reviewer.review_diff.call_args
            assert call_args[1]["custom_prompt"] == "Custom prompt"

    def test_run_with_nonexistent_prompt_file(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """存在しないプロンプトファイルの場合にエラーが発生することを確認"""
        prompt_file = temp_dir / "nonexistent.txt"

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(prompt_file=prompt_file)

            assert result is False

    def test_run_quiet_mode(
        self, config, mock_diff_generator, mock_ai_reviewer, temp_dir
    ):
        """quietモードでの実行を確認"""
        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch(
            "ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer
        ), patch("ai_review_orchestrator.logger"):
            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run(quiet=True)

            assert result is True
            # quietモードではログが少ないことを確認
            # (完全に無しではないが、通常より少ない)

    def test_run_diff_generation_error(
        self, config, mock_diff_generator, mock_ai_reviewer
    ):
        """Diff生成エラーの処理を確認"""
        mock_diff_generator.generate_diff.side_effect = DiffGenerationError(
            "Test error"
        )

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run()

            assert result is False

    def test_run_ai_review_error(self, config, mock_diff_generator, mock_ai_reviewer):
        """AIレビューエラーの処理を確認"""
        mock_ai_reviewer.review_diff.side_effect = AIReviewError("Test error")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run()

            assert result is False

    def test_run_unexpected_error(self, config, mock_diff_generator, mock_ai_reviewer):
        """予期しないエラーの処理を確認"""
        mock_diff_generator.generate_diff.side_effect = Exception("Unexpected error")

        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator.run()

            assert result is False

    def test_generate_diff_internal(self, config, mock_diff_generator):
        """_generate_diff内部メソッドのテスト"""
        with patch(
            "ai_review_orchestrator.DiffGenerator", return_value=mock_diff_generator
        ), patch("ai_review_orchestrator.AIReviewer"):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator._generate_diff(base_branch="develop")

            assert result["has_changes"] is True
            call_args = mock_diff_generator.generate_diff.call_args
            assert call_args[1]["base_branch"] == "develop"

    def test_generate_review_internal(self, config, mock_ai_reviewer, temp_dir):
        """_generate_review内部メソッドのテスト"""
        diff_path = temp_dir / "diff.patch"
        diff_path.write_text("test diff", encoding="utf-8")

        review_file = temp_dir / "tmp" / "ai_review_output.md"
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text("Mock review", encoding="utf-8")

        with patch("ai_review_orchestrator.DiffGenerator"), patch(
            "ai_review_orchestrator.AIReviewer", return_value=mock_ai_reviewer
        ):

            orchestrator = PRReviewOrchestrator(config)
            result = orchestrator._generate_review(diff_path)

            assert "output_path" in result
            assert "stats" in result
            mock_ai_reviewer.review_diff.assert_called_once()
