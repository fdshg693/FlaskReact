"""
Pytest configuration and shared fixtures

各テストで共通的に使用するfixtureを定義
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from git import Repo


@pytest.fixture
def temp_dir():
    """一時ディレクトリを作成するフィクスチャ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env(monkeypatch):
    """環境変数をモックするフィクスチャ"""
    env_vars = {
        "OPENAI_API_KEY": "test-api-key-12345",
        "AI_MODEL": "gpt-4o",
        "MAX_TOKENS": "10000",
        "TEMPERATURE": "0.1",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_git_repo(temp_dir, monkeypatch):
    """Gitリポジトリをモックするフィクスチャ"""
    # 一時ディレクトリにGitリポジトリを作成
    repo_path = temp_dir / "test_repo"
    repo_path.mkdir()

    repo = Repo.init(repo_path)

    # 初期コミットを作成
    test_file = repo_path / "test.txt"
    test_file.write_text("initial content")
    repo.index.add(["test.txt"])
    repo.index.commit("Initial commit")

    # リモートをモック
    mock_origin = MagicMock()
    mock_origin.fetch = MagicMock()
    repo.remote = MagicMock(return_value=mock_origin)

    return repo


@pytest.fixture
def sample_diff_content():
    """サンプルのdiffコンテンツを返すフィクスチャ"""
    return """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,8 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
+
+def goodbye():
+    print("Goodbye!")

if __name__ == "__main__":
     hello()
+    goodbye()
"""


@pytest.fixture
def sample_review_content():
    """サンプルのレビューコンテンツを返すフィクスチャ"""
    return """# Code Review

## Summary
The changes add a new function and update an existing one.

## Issues Found
None

## Suggestions
- Consider adding docstrings to new functions
- Add type hints for better code clarity
"""
