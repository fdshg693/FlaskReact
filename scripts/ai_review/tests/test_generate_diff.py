"""
generate_diff.py の単体テスト

DiffGeneratorクラスのメソッドをテストする
"""

from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError, InvalidGitRepositoryError

from config import ReviewConfig
from generate_diff import DiffGenerationError, DiffGenerator


class TestDiffGenerator:
    """DiffGeneratorクラスのテスト"""

    @pytest.fixture
    def config(self, mock_env, temp_dir, monkeypatch):
        """テスト用のReviewConfigを作成"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        return ReviewConfig()

    @pytest.fixture
    def mock_repo(self):
        """モックのGitリポジトリを作成"""
        repo = MagicMock()
        repo.git.diff = MagicMock(return_value="mock diff content")

        # remoteのモック
        mock_origin = MagicMock()
        mock_origin.fetch = MagicMock()
        repo.remote = MagicMock(return_value=mock_origin)

        # refsのモック
        mock_ref = MagicMock()
        mock_ref.name = "origin/main"
        mock_origin.refs = [mock_ref]

        return repo

    def test_init_with_valid_repo(self, config):
        """正常なGitリポジトリで初期化できることを確認"""
        with patch("generate_diff.Repo") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo

            generator = DiffGenerator(config)

            assert generator.config == config
            assert generator.repo == mock_repo

    def test_init_with_invalid_repo(self, config):
        """Gitリポジトリでない場合にエラーが発生することを確認"""
        with patch("generate_diff.Repo", side_effect=InvalidGitRepositoryError):
            with pytest.raises(
                DiffGenerationError, match="Gitリポジトリが見つかりません"
            ):
                DiffGenerator(config)

    def test_determine_base_branch_explicit(self, config, mock_repo):
        """明示的に指定されたブランチが優先されることを確認"""
        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            base_branch = generator._determine_base_branch("develop")

            assert base_branch == "develop"

    def test_determine_base_branch_env_pr_base_ref(
        self, config, mock_repo, monkeypatch
    ):
        """環境変数PR_BASE_REFが使用されることを確認"""
        monkeypatch.setenv("PR_BASE_REF", "staging")

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            base_branch = generator._determine_base_branch()

            assert base_branch == "staging"

    def test_determine_base_branch_env_input_target(
        self, config, mock_repo, monkeypatch
    ):
        """環境変数INPUT_TARGETが使用されることを確認"""
        monkeypatch.setenv("INPUT_TARGET", "feature")

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            base_branch = generator._determine_base_branch()

            assert base_branch == "feature"

    def test_determine_base_branch_default(self, config, mock_repo):
        """デフォルトで'main'が使用されることを確認"""
        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            base_branch = generator._determine_base_branch()

            assert base_branch == "main"

    def test_fetch_base_branch_success(self, config, mock_repo):
        """ブランチのフェッチが成功することを確認"""
        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            # エラーが発生しないことを確認
            generator._fetch_base_branch("main")

            mock_repo.remote.assert_called_once_with("origin")
            mock_repo.remote().fetch.assert_called_once_with("main")

    def test_fetch_base_branch_failure(self, config, mock_repo):
        """ブランチのフェッチに失敗した場合にエラーが発生することを確認"""
        mock_repo.remote().fetch.side_effect = GitCommandError("fetch", "error")

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            with pytest.raises(DiffGenerationError, match="フェッチに失敗しました"):
                generator._fetch_base_branch("nonexistent")

    def test_generate_diff_content_success(
        self, config, mock_repo, sample_diff_content
    ):
        """diff生成が成功することを確認"""
        mock_repo.git.diff.return_value = sample_diff_content

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            diff_content = generator._generate_diff_content("main")

            assert diff_content == sample_diff_content
            mock_repo.git.diff.assert_called_once_with(
                "origin/main...HEAD",
                unified=3,
                no_color=True,
                ignore_space_change=True,
            )

    def test_generate_diff_content_failure(self, config, mock_repo):
        """diff生成に失敗した場合にエラーが発生することを確認"""
        mock_repo.git.diff.side_effect = GitCommandError("diff", "error")

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            with pytest.raises(DiffGenerationError, match="diff生成に失敗しました"):
                generator._generate_diff_content("main")

    def test_generate_diff_with_changes(
        self, config, mock_repo, sample_diff_content, temp_dir
    ):
        """変更がある場合のdiff生成を確認"""
        mock_repo.git.diff.return_value = sample_diff_content
        output_path = temp_dir / "test_diff.patch"

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            result = generator.generate_diff(
                base_branch="main", output_path=output_path
            )

            assert result["has_changes"] is True
            assert result["diff_path"] == output_path
            assert result["line_count"] > 0
            assert result["base_branch"] == "main"
            assert output_path.exists()
            assert output_path.read_text(encoding="utf-8") == sample_diff_content

    def test_generate_diff_without_changes(self, config, mock_repo, temp_dir):
        """変更がない場合のdiff生成を確認"""
        mock_repo.git.diff.return_value = ""
        output_path = temp_dir / "test_diff.patch"

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            result = generator.generate_diff(
                base_branch="main", output_path=output_path
            )

            assert result["has_changes"] is False
            assert result["diff_path"] == output_path
            assert result["line_count"] == 0
            assert result["base_branch"] == "main"

    def test_generate_diff_default_output_path(
        self, config, mock_repo, sample_diff_content
    ):
        """デフォルトの出力パスが使用されることを確認"""
        mock_repo.git.diff.return_value = sample_diff_content

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            result = generator.generate_diff(base_branch="main")

            assert result["diff_path"] == config.get_diff_path()
            assert result["diff_path"].exists()

    def test_generate_diff_creates_output_dir(
        self, config, mock_repo, sample_diff_content, temp_dir
    ):
        """出力ディレクトリが自動作成されることを確認"""
        mock_repo.git.diff.return_value = sample_diff_content
        output_path = temp_dir / "subdir" / "test_diff.patch"

        with patch("generate_diff.Repo", return_value=mock_repo):
            generator = DiffGenerator(config)

            generator.generate_diff(base_branch="main", output_path=output_path)

            assert output_path.parent.exists()
            assert output_path.exists()
