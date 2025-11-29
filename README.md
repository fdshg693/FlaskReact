# 画像解析 / 機械学習 / LLM をブラウザから簡単操作する実験プロジェクト

- Backend: Flask (起動ランチャ `run_app.py`)
- Frontend: React (CDN, 直接JS/JSX) + 最小構成
- 補助UI: Streamlit アプリ (単機能のテスト・実験用)

## 📚 ドキュメント

詳細なドキュメントは以下を参照してください：

- **[プロジェクト概要](docs/project_overview/00_INDEX.md)**: プロジェクトの全体像、環境構築、実行方法
- **[開発規約](docs/dev_contract/INDEX.md)**: コーディング規約、テスト戦略、Git運用
- **[技術ナレッジ](docs/tech_knowledge/)**: Python、React、ツールの詳細仕様
- **[課題管理](docs/problems/INDEX.md)**: 既知の問題、TODO管理

## セットアップと実行

### 依存関係のインストール
```bash
uv sync
```

### 設定ファイル編集
- APIキー
    - `.env.example`を参考にして`.env`を作成
- 機械学習周りの設定YAML
    - `ml`フォルダ配下にあるYAML.EXAMPLEファイルを参考に（コピーでも可）YAMLファイルを作成
- VSCODE設定
    - `.vscode/**.json.example`を参考にして(コピーでも可) `.vscode/**.json`を作成

### アプリケーションの実行 (Flask)
```bash
uv run src/server/run_app.py  # http://localhost:8000
```

### Streamlit アプリ（任意）
```bash
uv run streamlit run src/server/streamlit/agent_app.py --server.port=8501
```

### Python環境 / 主なツール
- Python 3.13 + uv (依存管理 / 実行)
- 型/品質: mypy, ruff, pytest, pre-commit
- ログ/検証: loguru, pydantic, pathlib パターン徹底
- LLM/周辺: langchain など

### テスト
```bash
uv run pytest -q
```

### Github Actions / 自動化
- PR 時に AI レビュー ワークフロー実行（手動トリガも可）・`scripts`配下のスクリプト実行してAIレビューをローカルで行うことも可能
- `pre-commit` は `.pre-commit-config.yaml` を参照（フック導入 `uv run pre-commit install`）
- Github Copilotを活用するプロンプト・指示・エージェント設定が`.github`フォルダ配下に格納済み