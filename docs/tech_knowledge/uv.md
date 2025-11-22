# UVの使い方

## 公式ドキュメント
- tool
    - https://docs.astral.sh/uv/guides/tools/#installing-tools
- プロジェクト構造
    - https://docs.astral.sh/uv/guides/projects/
- インストール方法
    - https://docs.astral.sh/uv/getting-started/installation/
- コマンド一覧
    - https://docs.astral.sh/uv/getting-started/features/ 


### 既にインストールされているPYTHONをUVで管理する
- uv python list --no-managed-python;

### スクリプトの実行
- uv run *.py
    - 実行する際に仮想環境が自動でアクティブになります。

### パッケージの追加
- uv add <package_name>
    - mypy, flaskなどのパッケージを追加する場合は、`uv add`コマンドを使用します。
    - project.tomlに追加されます。
    - uv syncを実行することで、仮想環境にインストールされます。
### パッケージのアンインストール
- uv remove <package_name>
### パッケージの確認
- uv pip list
    - 現在のプロジェクトでインストールされているパッケージを確認できます。

### ツールの追加
- uv add tool <tool_name>
    - ruff, blackなどのツールを追加する場合は、`uv add tool`コマンドを使用します。
        - 今回はruffをパッケージとして追加したので注意！！
    - project.tomlに追加されません。
    - uv syncを実行しても追加されないので、手動でインストールする必要があります。
### ツールの追加（開発用）
- uv add tool --dev <tool_name>
    - 開発用のツールを追加する場合は、`--dev`オプションを使用します。
    - 開発用のツールは、プロジェクトの開発環境でのみ使用されます。

### ツールのアンインストール
uv tool uninstall <tool_name>
### ツールの確認
- uv tool list
    - 現在のプロジェクトでインストールされているツールを確認できます。
### ツールの使い方
- <tool_name> <command>
    - 例えば、`ruff check .`のように実行します。
    - ツールをインストールした後は、コマンドを直接実行できます。

### ツールをインストールせずに実行
- uvx <command>
    - 例えば、`uvx ruff check .`のように実行します。
    - ツールをインストールせずに実行できます。

## 依存関係の確認
- uv sync


## パス
### ツールをパスに追加
- warning: `/Users/seiwan/.local/bin` is not on your PATH. To use installed tools, run `export PATH="/Users/seiwan/.local/bin:$PATH"` or `uv tool update-shell`.
- echo 'export PATH="/Users/seiwan/.local/bin:$PATH"' >> ~/.zshrc
- source ~/.zshrc

### pyproject.toml
```toml
[tool.uv]
name = "FlaskReact"
version = "0.1.0"
description = "A Flask and React project"
authors = ["seiwan <seiwan@example.com>"]  
license = "MIT"
python = ">=3.13"
[tool.uv.dependencies]
mypy = ">=0.991"
ruff = ">=0.7"
[tool.uv.dev-dependencies]
black = ">=23.0"
uv = ">=0.7"
[tool.uv.tools]
ruff = { version = ">=0.7", dev = true }
black = { version = ">=23.0", dev = true }
```