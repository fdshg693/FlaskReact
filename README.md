# 画像解析 / 機械学習 / LLM をブラウザから簡単操作する実験プロジェクト

- Backend: Flask (起動ランチャ `run_app.py`)
- Frontend: React (CDN, 直接JS/JSX) + 最小構成
- 補助UI: Streamlit アプリ (`src/streamlit/`)

## セットアップと実行

### 依存関係のインストール
```bash
uv sync
```

### アプリケーションの実行 (Flask)
```bash
uv run run_app.py  # http://localhost:8000
```
> 重要: `server/app.py` を直接実行せず必ず `run_app.py` を使う。Python Path や絶対インポートが自動設定されるため。

### Streamlit アプリ（任意）
```bash
uv run streamlit run src/streamlit/agent_app.py --server.port=8501
```
他: `machine_learning.py`, `simple_app.py` も同ディレクトリにあり。

### Python環境 / 主なツール
- Python 3.13 + uv (依存管理 / 実行)
- 型/品質: mypy, ruff, pytest, pre-commit
- ログ/検証: loguru, pydantic, pathlib パターン徹底
- LLM/周辺: langchain など
- サンプル環境変数: `sample.env` を `.env` にコピーして編集

### テスト
```bash
uv run pytest -q
```

### Github Actions / 自動化
- PR 時に AI レビュー ワークフロー実行（手動トリガも可）
- `pre-commit` は `.pre-commit-config.yaml` を参照（フック導入 `uv run pre-commit install`）


## Github Copilot / Prompts
詳細な使い方は `.github/explanation.md` を参照。VSCode Chat で `/` から各プロンプトを呼び出し、`instructions` 系はコードと一緒にコンテキストへ含める。

### 現状概要
- ML: Iris などの学習 (`uv run python -m src.machineLearning.ml_class`) → モデル/スケーラは将来的に保存ディレクトリ統合予定
- LLM: 画像解析 / PDF テキスト抽出 / テキスト分割 関数 (`src/llm/`)
- API: 画像解析 `/api/analyze-image`, PDF 抽出 `/api/extract-pdf-text`, テキスト分割 `/api/split-text`, Iris 予測 `/api/iris-prediction`
- Frontend: Iris 単発 & CSV バッチ予測、画像・PDF・テキスト操作 UI
- Streamlit: LLM エージェント実験 UI (`agent_app.py` 等)
## 全体フォルダ構成（抜粋）
```
FlaskReact/
├── src/
│   ├── run_app.py        # 起動ランチャ
│   ├── server/           # Flask エントリ & ルーティング
│   ├── llm/              # 画像解析 / PDF / テキスト分割
│   ├── machineLearning/  # 学習 & 推論コード
│   ├── streamlit/        # 実験用 UI
│   ├── util/             # 共通ユーティリティ
│   └── scrape/           # スクレイピング
├── static/               # React (CDN) フロント
├── data/                 # 入力データ / サンプル
├── csvLog / curveLog     # 学習ログ & 曲線画像
├── tests/                # pytest テスト
├── docs/                 # 補助ドキュメント
└── sample.env            # 環境変数サンプル


---
今後: モデル保存場所の統一 / LLM 機能追加 / API テスト自動化 などを拡張予定。
