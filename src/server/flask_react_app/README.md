# Server Module

Flask アプリケーションのサーバーコアモジュール。アプリケーションの初期化、設定管理、ページルーティングを担当します。

## 📁 ファイル構成

```
src/server/
├── app.py          # Flask アプリケーションのエントリーポイントと初期化
├── config.py       # 環境変数ベースの設定管理
├── pages.py        # 静的ページのルーティング定義
└── api/            # API エンドポイント群（別ディレクトリ）
    ├── image.py    # 画像処理 API
    ├── iris.py     # 機械学習 API
    ├── pdf.py      # PDF 処理 API
    └── text.py     # テキスト処理 API
```

---

## 📄 ファイル詳細

### `app.py`

Flask アプリケーションのメインファイル。アプリケーションファクトリパターンを採用し、統一的なエラーハンドリングとセキュリティ機能を提供します。

#### 重要な関数

##### `validate_file_upload(file, allowed_extensions, max_size_mb=10) -> str`

アップロードされたファイルのバリデーションを実行します。

**検証項目:**

- ファイルの存在確認
- 拡張子のホワイトリスト検証
- ファイルサイズ制限チェック（デフォルト: 10MB）
- セキュアなファイル名の生成（パストラバーサル攻撃対策）

**引数:**

- `file`: `werkzeug.datastructures.FileStorage` オブジェクト
- `allowed_extensions`: 許可する拡張子のセット（例: `{'jpg', 'png', 'pdf'}`）
- `max_size_mb`: 最大ファイルサイズ（MB 単位）

**戻り値:**

- `str`: `secure_filename()` で処理された安全なファイル名

**例外:**

- `ValueError`: ファイル未選択、不正な拡張子、サイズ超過時

**TODO:**

- このユーティリティを `util/file_validation.py` に移動し、責任の分離を図る
- ファイルサイズチェックをポインタ操作ではなくライブラリで実装

##### `handle_api_errors(f)`

API エンドポイント用の統一エラーハンドリングデコレータ。

**処理される例外:**

- `ValueError` → 400 Bad Request（バリデーションエラー）
- `FileNotFoundError` → 500 Internal Server Error（ファイル処理エラー）
- `Exception` → 500 Internal Server Error（予期しないエラー）

**使用例:**

```python
@app.route('/api/example')
@handle_api_errors
def example_endpoint():
    # エラーハンドリングは自動処理される
    return jsonify({"result": "success"})
```

##### `create_app() -> Flask`

Flask アプリケーションファクトリ関数。**直接実行せず、必ず `uv run run_app.py` 経由で起動してください。**

**初期化処理:**

1. 静的ファイルディレクトリの設定（CDN React 対応）
2. 環境変数からの設定読み込み（`get_settings()` 経由）
3. CORS 設定の適用
4. Blueprint の登録:
   - `pages_bp`: ページルーティング
   - `text_bp`, `iris_bp`, `pdf_bp`, `image_bp`: API エンドポイント
5. 統一エラーハンドラの登録（400, 404, 500）

**起動方法:**

```bash
uv run run_app.py  # ✅ 推奨（Python path が自動設定される）
```

**直接実行しない理由:**
`run_app.py` がプロジェクトルートを Python path に追加するため、絶対インポート（`from llm.image import ...`）が正しく動作します。

---

### `config.py`

Pydantic Settings を使用した型安全な環境変数管理。

#### `Settings` クラス

アプリケーション全体の設定を一元管理します。

**環境変数プレフィックス:** `FLASKREACT_`  
**設定ファイル:** `.env`（オプション、起動時に `load_dotenv_workspace()` で読み込み）

**設定項目:**

| 属性                       | 型          | デフォルト値                            | 説明                       |
| -------------------------- | ----------- | --------------------------------------- | -------------------------- |
| `cors_origins`             | `List[str]` | `["http://localhost:3000", ...]`        | CORS 許可オリジン          |
| `allowed_image_extensions` | `Set[str]`  | `{"png", "jpg", "jpeg", "gif"}`         | 画像ファイル許可拡張子     |
| `allowed_pdf_extensions`   | `Set[str]`  | `{"pdf"}`                               | PDF ファイル許可拡張子     |
| `max_image_size_mb`        | `int`       | `5`                                     | 画像最大サイズ（MB）       |
| `max_pdf_size_mb`          | `int`       | `10`                                    | PDF 最大サイズ（MB）       |
| `app_root`                 | `Path`      | `PATHS.src`                             | アプリケーションルート     |
| `model_path`               | `Path`      | `PATHS.ml_outputs/param/models_*.pth`   | 機械学習モデルパス         |
| `scaler_path`              | `Path`      | `PATHS.ml_outputs/scaler/scaler.joblib` | データスケーラーパス       |
| `checkpoint_path`          | `Path`      | `PATHS.ml_outputs/checkpoints/...`      | モデルチェックポイントパス |

**使用例:**

```python
from server.flask_react_app.config import get_settings

settings = get_settings()
print(settings.max_image_size_mb)  # 5
```

**環境変数での上書き例（`.env` ファイル）:**

```env
FLASKREACT_MAX_IMAGE_SIZE_MB=10
FLASKREACT_CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

**TODO:**

- 起動時にモデルファイルの存在チェックを追加
- ファイルが存在しない場合のフォールバック処理を実装
- モデルパスのベタ書きを改善（環境変数化 or 最新モデル自動検出 or YAML/JSON 設定）

---

### `pages.py`

静的 HTML ページのルーティングを定義する Blueprint。CDN React で構築されたフロントエンドページを配信します。

#### ルートエンドポイント

##### `GET /`

ルートパスアクセス時にホームページを返します。  
**配信ファイル:** `static/home/index.html`

##### `GET /home`

Iris 予測機能（単一・バッチ）を提供するホームページ。  
**配信ファイル:** `static/home/index.html`

##### `GET /csvTest`

CSV ファイルのアップロードと処理をテストするページ。  
**配信ファイル:** `static/csvTest/index.html`

##### `GET /image`

画像分析・PDF 処理・テキスト分割などの LLM 機能を提供するページ。  
**配信ファイル:** `static/image/index.html`

##### `GET /data/<path:path>`

`static/data` 配下の静的ファイルを配信します。

**セキュリティ対策:**

- ディレクトリトラバーサル攻撃を防止（`..`, `/`, `\` を拒否）
- パスの正規化と検証（`data` ディレクトリ外へのアクセス禁止）

**エラーレスポンス:**

- `400`: 不正なパス形式
- `403`: アクセス権限なし
- `404`: ファイルが見つからない

**使用例:**

```
http://localhost:8000/data/sample.csv  # ✅ OK
http://localhost:8000/data/../secret   # ❌ 400 エラー
```

---

## 🚀 起動方法

### 推奨起動コマンド

```bash
# 依存関係のインストール
uv sync

# アプリケーション起動（Python path が自動設定される）
uv run run_app.py
```

### デバッグモード起動

`app.py` の `__main__` ブロックから直接起動することも可能ですが、絶対インポートが動作しない可能性があるため推奨されません。

---

## 🔧 開発ガイド

### 新しい API エンドポイントの追加

1. `src/server/api/` に新しいモジュールを作成
2. Blueprint を定義してエンドポイントを実装
3. `app.py` の `create_app()` で Blueprint を登録

**例: `api/example.py`**

```python
from flask import Blueprint, jsonify
from server.app import handle_api_errors

example_bp = Blueprint("example", __name__)

@example_bp.route("/api/example", methods=["POST"])
@handle_api_errors
def example_endpoint():
    return jsonify({"message": "Hello, World!"})
```

**`app.py` への登録:**

```python
from server.api.example import example_bp

def create_app() -> Flask:
    # ...
    app.register_blueprint(example_bp)
    # ...
```

### 新しいページの追加

1. `static/<page_name>/` にディレクトリを作成
2. `index.html` と必要な JavaScript ファイルを配置
3. `pages.py` に新しいルートを追加

**例:**

```python
@pages_bp.route("/newpage")
def serve_new_page() -> Response:
    new_page_static_path = PATHS.static / "newpage"
    return send_from_directory(str(new_page_static_path), "index.html")
```

---

## ⚠️ 注意事項

1. **起動方法:** run_app.py で F5 を押してデバッグ実行してください
2. **セキュリティ:** ファイルアップロードは `validate_file_upload()` で必ずバリデーションを実行
3. **エラーハンドリング:** API エンドポイントには `@handle_api_errors` デコレータを適用
4. **CORS 設定:** 本番環境では `cors_origins` を適切に制限してください

---

## 📚 関連ドキュメント

- プロジェクト全体の説明: `README.md` (プロジェクトルート)
- API 仕様: `src/server/api/` の各モジュール
- 開発ガイドライン: `.github/copilot-instructions.md`
- Modern Python パターン: `.github/instructions/modern.instructions.md`
