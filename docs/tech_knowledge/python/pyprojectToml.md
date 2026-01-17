# pyproject.tomlの書き方（モダンな形式）

## 基本構造

PyPA（Python Packaging Authority）標準の`pyproject.toml`は、`uv`や`pip`などのツールに対応した設定ファイルです。

### 最小限の構成

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "プロジェクト説明"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.28.0",
    "pydantic>=2.0",
]
```

## 主要セクション

### 1. **[project]** - プロジェクト基本情報
```toml
[project]
name = "flaskreact"
version = "0.1.0"
description = "Flask + React + Streamlit for ML & LLM"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    { name = "Author Name", email = "author@example.com" }
]
license = { text = "MIT" }
keywords = ["flask", "react", "ml"]
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "flask>=2.0",
    "pydantic>=2.0",
    "loguru>=0.7",
]
```

### 2. **[project.optional-dependencies]** - オプション依存関係
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
ml = [
    "torch>=2.0",
    "scikit-learn>=1.3",
    "pandas>=2.0",
]
llm = [
    "openai>=1.0",
    "langchain>=0.1",
]
```

### 3. **[tool.uv]** - uvツール設定
```toml
[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "black>=23.0",
]
```

### 4. **[tool.pytest.ini_options]** - pytest設定
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
```

### 5. **[tool.ruff]** - Ruff（Linter）設定
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
]
ignore = ["E501", "W503"]
```

### 6. **[tool.mypy]** - mypy（型チェッカー）設定
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
ignore_missing_imports = true
```

## FlaskReactプロジェクトの実例

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "flaskreact"
version = "0.1.0"
description = "Experimental Flask + React + Streamlit for ML & LLM"
requires-python = ">=3.10"

dependencies = [
    "flask>=2.3",
    "pydantic>=2.0",
    "loguru>=0.7",
    "werkzeug>=2.3",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
ml = [
    "torch>=2.0",
    "scikit-learn>=1.3",
    "numpy>=1.24",
    "pandas>=2.0",
]
llm = [
    "openai>=1.0",
    "langchain>=0.1",
    "pypdf>=3.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
ignore_missing_imports = true
```

## 重要なポイント

| 項目 | 説明 |
|------|------|
| **requires-python** | Python最小バージョンを指定（>=3.10推奨） |
| **dependencies** | 必須の本番用パッケージ |
| **optional-dependencies** | `pip install package[dev]`でインストール可能なオプション |
| **dev-dependencies (uv)** | `uv sync`で自動インストールされるdev用パッケージ |
| **build-system** | ビルドシステムを明示的に指定 |

## コマンド例

```bash
# 依存関係を同期
uv sync

# 開発用パッケージも含める
uv sync --all-extras

# 特定のグループのみ
uv sync --only-group dev
```
