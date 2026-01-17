"""
config.py の単体テスト

ReviewConfigクラスの初期化、バリデーション、パス管理をテストする
"""

import os

import pytest

from config import ConfigurationError, ReviewConfig


class TestReviewConfig:
    """ReviewConfigクラスのテスト"""

    def test_init_with_valid_env(self, mock_env, temp_dir, monkeypatch):
        """正常な環境変数で初期化できることを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()

        assert config.openai_api_key == "test-api-key-12345"
        assert config.ai_model == "gpt-4o"
        assert config.max_tokens == 10000
        assert config.temperature == 0.1

    def test_init_without_api_key(self, monkeypatch, temp_dir):
        """API Keyがない場合にエラーが発生することを確認"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # .envファイルの読み込みを無効化
        monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)

        with pytest.raises(
            ConfigurationError, match="OPENAI_API_KEY が設定されていません"
        ):
            ReviewConfig()

    def test_init_with_invalid_max_tokens(self, mock_env, monkeypatch):
        """MAX_TOKENSが数値でない場合にエラーが発生することを確認"""
        monkeypatch.setenv("MAX_TOKENS", "invalid")

        with pytest.raises(
            ConfigurationError, match="MAX_TOKENS は整数である必要があります"
        ):
            ReviewConfig()

    def test_init_with_invalid_temperature(self, mock_env, monkeypatch):
        """TEMPERATUREが数値でない場合にエラーが発生することを確認"""
        monkeypatch.setenv("TEMPERATURE", "invalid")

        with pytest.raises(
            ConfigurationError, match="TEMPERATURE は数値である必要があります"
        ):
            ReviewConfig()

    def test_default_values(self, monkeypatch, temp_dir):
        """デフォルト値が正しく設定されることを確認"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        # AI_MODEL, MAX_TOKENS, TEMPERATUREを削除
        monkeypatch.delenv("AI_MODEL", raising=False)
        monkeypatch.delenv("MAX_TOKENS", raising=False)
        monkeypatch.delenv("TEMPERATURE", raising=False)

        config = ReviewConfig()

        assert config.ai_model == "gpt-4o"
        assert config.max_tokens == 10000
        assert config.temperature == 0.1

    def test_detect_project_root_with_env(self, mock_env, temp_dir, monkeypatch):
        """環境変数PROJECT_ROOTが優先されることを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()

        assert config.project_root == temp_dir

    def test_detect_project_root_with_git(self, mock_env, temp_dir, monkeypatch):
        """.gitディレクトリがある場合にルートが検出されることを確認"""

        # 現在のディレクトリを保存
        original_cwd = os.getcwd()

        try:
            # .gitディレクトリを作成（ファイルとして）
            git_file = temp_dir / ".git"
            # Windowsでのファイルロック問題を避けるため、ディレクトリではなくファイルとして作成
            git_file.touch()

            # PROJECT_ROOT環境変数を削除
            monkeypatch.delenv("PROJECT_ROOT", raising=False)

            # カレントディレクトリを変更
            monkeypatch.chdir(temp_dir)

            config = ReviewConfig()

            assert config.project_root == temp_dir
        finally:
            # カレントディレクトリを元に戻す
            os.chdir(original_cwd)

    def test_get_diff_path(self, mock_env, temp_dir, monkeypatch):
        """diff出力パスが正しく取得できることを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()
        diff_path = config.get_diff_path()

        assert diff_path.name == "diff.patch"
        assert diff_path.parent.name == "tmp"

    def test_get_review_output_path(self, mock_env, temp_dir, monkeypatch):
        """レビュー出力パスが正しく取得できることを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()
        review_path = config.get_review_output_path()

        assert review_path.name == "ai_review_output.md"
        assert review_path.parent.name == "tmp"

    def test_validate_success(self, mock_env, temp_dir, monkeypatch):
        """正常な設定でvalidateがTrueを返すことを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        temp_dir.mkdir(exist_ok=True)

        config = ReviewConfig()

        assert config.validate() is True

    def test_validate_invalid_max_tokens(self, mock_env, temp_dir, monkeypatch):
        """max_tokensが0以下の場合にvalidateがFalseを返すことを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        monkeypatch.setenv("MAX_TOKENS", "0")
        temp_dir.mkdir(exist_ok=True)

        config = ReviewConfig()

        assert config.validate() is False

    def test_validate_invalid_temperature(self, mock_env, temp_dir, monkeypatch):
        """temperatureが範囲外の場合にvalidateがFalseを返すことを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))
        monkeypatch.setenv("TEMPERATURE", "3.0")
        temp_dir.mkdir(exist_ok=True)

        config = ReviewConfig()

        assert config.validate() is False

    def test_tmp_dir_creation(self, mock_env, temp_dir, monkeypatch):
        """tmpディレクトリが自動作成されることを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()

        assert config.tmp_dir.exists()
        assert config.tmp_dir.is_dir()

    def test_repr(self, mock_env, temp_dir, monkeypatch):
        """__repr__が正しく動作することを確認"""
        monkeypatch.setenv("PROJECT_ROOT", str(temp_dir))

        config = ReviewConfig()
        repr_str = repr(config)

        assert "ReviewConfig" in repr_str
        assert "project_root" in repr_str
        assert "ai_model=gpt-4o" in repr_str
        assert "***" in repr_str  # API keyは隠される
        assert "test-api-key" not in repr_str  # 実際のキーは表示されない
