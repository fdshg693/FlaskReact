# 画像解析・機械学習などをWEBアプリ上で簡単に操作することが出来ることを目指す実験的なプロジェクト

- バックエンド　FLASK
- フロントエンド　REACT(CDN)

## セットアップと実行

### 依存関係のインストール
```bash
uv sync
```

### アプリケーションの実行
```bash
python run_app.py
```

アプリケーションは `http://localhost:8000` で起動します。

> **注意**: `server/app.py` を直接実行せず、必ず `run_app.py` を使用してください。
> これにより、Pythonパスが自動的に設定され、モジュールのインポートが正しく動作します。

### Python環境
- python 3.13を使用
- uvを使用して仮想環境を管理
    - uv syncでプロジェクトの依存関係をインストール
    - uv runでスクリプトを実行
- 開発にて使用しているツール
    - mypy: 型チェック
    - pydoc: ドキュメンテーション生成
- 取り入れた新しめのライブラリ
    - langchain: LLMを利用したアプリケーション開発のためのライブラリ
    - pydantic: データ検証と設定管理のためのライブラリ
    - uv: Pythonプロジェクトの依存関係管理ツール
    - loguru: ロギングのためのライブラリ
    - pathlib: パス操作のためのライブラリ
    - pytest: テストのためのライブラリ
    - ruff: コード静的解析ツール
- 開発にて取り入れたツール
    - pre-commit: コードの品質を保つためのツール

### Github Actions
- プルリクエストの際に、AIがコードをレビュー
    - プルリクエストの時以外にも、手動で実行可能
### pre-commit
- https://pre-commit.com/https://pre-commit.com/
- .pre-commit-config.yaml


# ファイル構成
## モジュール構成
### llm
- LLMを利用した関数を定義
    - `agent.py`
        - LLMを利用したエージェントの定義
    - `document_search_tools.py`
        - ドキュメント検索のためのツール
    - `function_calling.py`
        - 関数呼び出しのためのツール
    - `image.py`
        - 画像を生成AIに描写させるためのツール
    - `pdf.py`
        - PDFファイルを文字起こしするためのツール
    - `text_splitter.py`
        - テキストを分割するためのツール
## Github Copilot
- Instructions
    - `.github/instructions/modern.instructions.md`
        - pythonファイルに対して適用されて、現在のpython環境をAIに教える 
            - コンテキストにPythonファイルがある必要がある
- Prompts
    - ### 自動では反映されないので、「/」をチャットに入力してショートカットから選択する必要がある
    - `.github/prompts/fix.prompt.md`
        - レビューの内容に基づいて。最も重大な問題を修正するためのプロンプト
            - レビューのファイルが既に作成されている必要がある
            - コンテキストに、修正対象となるpythonファイルおよび、そのコードに対するレビューのファイルが必要 
        - 修正後に、レビューファイルの内容を更新するので、このプロンプトを繰り返し使用することが可能
    - .github/prompts/refine.prompt.md
        - プロンプトを更に洗練させるためのプロンプト
            - コンテキストにプロンプトを含める必要がある
    - .github/prompts/modernize_python.prompt.md
        - Pythonのコードに最新のスタイル・トレンドを取り入れる。pythonの勉強として、新しいライブラリを取り入れたり、最新のPythonの機能を使用するためのプロンプト
            - コンテキストにpythonファイルが必要

### 現状

- 実行方法
    - python server/app.py

- バックエンド
    - FLASKで単純なルーティング
    - IRISデータを機械学習させたモデルを保存して呼び出せるようにしたい（途中）    
    - LLMフォルダに生成AIなど（LANGCHAINが主）を利用した関数を定義
        - 画像評価
        - PDF文字起こし
        - テキスト分割　
- フロントエンド
    - アイリスのデータを入力すると、ランダムなアイリス種の回答を返却
        - 複数データをCSV形式で入力・アップロードも可
    - 画像をアップロードすると、AIが描写してくれる
    - PDFファイルをアップロードすると、AIが文字起こししてくれる
    - 文字入力すると、分割して返す
    -