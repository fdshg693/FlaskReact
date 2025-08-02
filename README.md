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
uv run run_app.py
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


## Github Copilot
- Instructions
    - `.github/instructions/modern.instructions.md`
        - pythonファイルに対して適用されて、現在のpython環境をAIに教える 
            - コンテキストにPythonファイルがある必要がある
    - `.github/instructions/react.instructions.md`
        - Reactのコードに対して適用されて、現在のReact環境をAIに教える
            - コンテキストにJSXファイルがある必要がある
- Prompts
    - ### 自動では反映されないので、「/」をチャットに入力してショートカットから選択する必要がある
    #### Python
    - `.github/prompts/fix.prompt.md`
        - レビューの内容に基づいて。最も重大な問題を修正するためのプロンプト
            - レビューのファイルが既に作成されている必要がある
            - コンテキストに、修正対象となるpythonファイルおよび、そのコードに対するレビューのファイルが必要 
        - 修正後に、レビューファイルの内容を更新するので、このプロンプトを繰り返し使用することが可能
    - `.github/prompts/refine.prompt.md`
        - プロンプトを更に洗練させるためのプロンプト
            - コンテキストにプロンプトを含める必要がある
    - `.github/prompts/modernize_python.prompt.md`
        - Pythonのコードに最新のスタイル・トレンドを取り入れる。pythonの勉強として、新しいライブラリを取り入れたり、最新のPythonの機能を使用するためのプロンプト
            - コンテキストにpythonファイルが必要
    - `.github/prompts/review_python.prompt.md`
        - Pythonのコードをレビューするためのプロンプト
            - コンテキストにpythonファイルが必要
            - `review`フォルダをコードと同じ階層に作成し、そこにレビューの内容を保存する
            - ここの作生物は、上の`fix.prompt.md`にて読み取り・書き込みが行われる
    #### React
    - `.github/prompts/react_fix.prompt.md`
        - Reactのコードに対して、最も重大な問題を修正するためのプロンプト
            - レビューのファイルが既に作成されている必要がある
            - コンテキストに、修正対象となるJSXファイルおよび、そのコードに対するレビューのファイルが必要
        - 修正後に、レビューファイルの内容を更新するので、このプロンプトを繰り返し使用することが可能
    - `.github/prompts/react_review.prompt.md`
        - Reactのコードをレビューするためのプロンプト
            - コンテキストにJSXファイルが必要
            - `review`フォルダをコードと同じ階層に作成し、そこにレビューの内容を保存する
            - ここの作生物は、上の`react_fix.prompt.md`にて読み取り・書き込みが行われる

### 現状

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
## 全体フォルダ構成
```
FlaskReact/
├── .github/
│   ├── instructions/ # AIにコードのコンテキストを教えるためのファイル
│   ├──  prompts/ # AIにコードを修正・レビュー等を依頼するためのプロンプト
│   └── workflows/ # Github Actionsの設定ファイル
├── .vscode/ # VSCodeの設定ファイル
├── csvLog # 機械学習の学習ログを記録したCSVファイル
├── curveLog/ # 機械学習の精度記録を画像化したファイル
├── data/ # 様々な箇所に利用するデータを格納するフォルダ　詳しくは`data/README.md`を参照
├── docs/ # ライブラリ等の使い方をメモしたドキュメント
├── experiment/ # 様々な実験を行うフォルダ
├── llm/ # LLMを利用した関数を定義　詳しくは`llm/README.md`を参照
├── logs/ # ログを保存するフォルダ
├── machineLearning/ # 機械学習関連のコード
├── param/ # 機械学習モデルのパラメータを保存するフォルダ
├── scaler/ # 機械学習モデルのスケーラーを保存するフォルダ
├── scrape/ # スクレイピング関連のコード
├── server/ # Flaskのサーバーコード
├── static/ # 静的ファイル（CSS, JS, 画像等）
│   ├── csvTest/ # CSVファイルのテスト用
│   ├── home/ # IRISの１データおよび、複数データの判定を可能とするページ
│   ├── image/ # 画像の描画AI・PDF文字起こし・テキスト分割を可能とするページ
├── test/ # テスト用のページ
├── util/ # ユーティリティ関数を定義
