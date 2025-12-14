from __future__ import annotations

from functools import wraps

from flask import Flask, jsonify
from flask_cors import CORS
from loguru import logger
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import PATHS
from server.flask_react_app.api import image_bp, iris_bp, pdf_bp, text_bp
from server.flask_react_app.config import Settings
from server.flask_react_app.pages import pages_bp


# TODO: validate_file_uploadを util/file_validation.py に移動し、責任の分離を図る
def validate_file_upload(
    file: FileStorage, allowed_extensions: set[str], max_size_mb: int = 10
) -> str:
    """アップロードされたファイルのセキュリティとサイズ制約を検証する。

    Args:
        file: Werkzeugのアップロードファイルオブジェクト
        allowed_extensions: 許可する拡張子のセット（例: {'jpg', 'png', 'pdf'}）
        max_size_mb: 許可する最大ファイルサイズ（MB単位、デフォルト: 10MB）

    Returns:
        str: セキュリティで保護されたファイル名（secure_filenameで処理済み）

    Raises:
        ValueError: ファイルが未選択、拡張子が不正、またはサイズ超過の場合
    """
    # ファイルの存在チェック: ファイルが選択されているか確認
    if not file or file.filename == "":
        raise ValueError("No file selected")

    # 拡張子のバリデーション: 許可された拡張子かチェック
    # ファイル名に"."が含まれているか、拡張子が許可リストにあるかを検証
    if (
        "." not in file.filename
        or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions
    ):
        raise ValueError(f"File type not allowed. Allowed: {allowed_extensions}")

    # TODO: ファイルサイズのチェックをポインタではなく何かしらのライブラリで行えるようにする
    # ファイルサイズのチェック
    # seek(0, 2)でファイルの末尾に移動し、tell()で現在位置（＝ファイルサイズ）を取得
    file.seek(0, 2)  # ファイルポインタを末尾に移動
    file_size = file.tell()  # 現在位置（バイト数）を取得
    file.seek(0)  # ファイルポインタを先頭に戻す（後続処理のため）

    # サイズ上限を超えている場合はエラー（MB → バイトに変換して比較）
    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")

    # 4. セキュアなファイル名を生成して返す
    # secure_filenameでパストラバーサル攻撃などを防止
    # ファイル名が空の場合は "unnamed_file" を使用
    return secure_filename(file.filename) or "unnamed_file"


def handle_api_errors(f):
    """APIエンドポイントの共通エラーハンドリングを行うデコレータ。

    このデコレータを適用することで、各エンドポイント関数内で個別に
    エラーハンドリングを記述する必要がなくなります。

    処理される例外とHTTPステータスコード:
    - ValueError: バリデーションエラー → 400 Bad Request
    - FileNotFoundError: ファイル処理エラー → 500 Internal Server Error
    - その他のException: 予期しないエラー → 500 Internal Server Error

    使用例:
        @app.route('/api/example')
        @handle_api_errors
        def example_endpoint():
            # エラーハンドリングはデコレータが自動的に処理
            return jsonify({"result": "success"})

    Args:
        f: デコレートする関数(通常はFlaskのエンドポイント関数)

    Returns:
        ラップされた関数。元の関数の戻り値、またはエラー時のJSON応答を返す
    """

    @wraps(f)  # 元の関数のメタデータ(__name__, __doc__等)を保持
    def decorated_function(*args, **kwargs):
        try:
            # ラップされた関数を実行
            return f(*args, **kwargs)
        except ValueError as e:
            # バリデーションエラー: リクエストデータの形式が不正な場合
            # (例: 必須パラメータ不足、不正な値、ファイル形式エラー等)
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid input data"}), 400
        except FileNotFoundError as e:
            # ファイル処理エラー: アップロードファイルの処理中にエラーが発生
            # (例: 一時ファイルの削除失敗、想定外のファイルパス等)
            logger.error(f"File error in {f.__name__}: {e}")
            return jsonify({"error": "File processing error"}), 500
        except Exception as e:
            # その他の予期しないエラー: 上記以外の全ての例外をキャッチ
            # (例: 外部API通信エラー、メモリ不足、未定義変数参照等)
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return decorated_function


def create_app() -> Flask:
    """Flaskアプリケーションファクトリ関数。

    プロジェクトの推奨起動パターンに準拠したアプリケーション初期化を実行します。
    注意: この関数を直接呼び出さず、必ず `uv run run_app.py` を使用してください。

    主な処理内容:
    - Settings()からCORS設定とファイルサイズ制限を読み込み
    - 静的ファイル配信の設定(CDN React用)
    - Blueprintの登録:
        * pages_bp: ページルーティング(/, /image など)
        * text_bp: テキスト処理API(/api/split-text など)
        * iris_bp: 機械学習API(/api/iris-prediction など)
        * pdf_bp: PDF処理API(/api/extract-pdf-text など)
        * image_bp: 画像処理API(/api/analyze-image など)
    - 統一エラーハンドラの追加(400, 404, 500)

    Returns:
        Flask: 設定済みのFlaskアプリケーションインスタンス

    Note:
        この関数は run_app.py から呼び出されることで、
        プロジェクトルートがPython pathに追加され、
        絶対インポート(from llm.image import ...)が正しく動作します。
    """

    # src/server/config/paths.pyから該当パスのインスタンスを生成
    static_dir = PATHS.flask_static
    # Flaskクラスの引数の意味[__name__: モジュール名。Flaskはこれを使ってリソースの場所を特定する。][#static_folder: staticのディレクトリパス。][#static_url_path: staticにアクセスするためのURLパス。]
    # static_url_pathを空文字列にすることで、各ページディレクトリ配下のアセット(home/js/App.js等)を直接参照可能にする
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="")
    # .envファイルからFLASKREACT_プレフィックス付きの環境変数を自動読み込み（実際には"FLASKREACT_"プレフィックスが存在していないため、何も渡されていない）
    settings = Settings()

    # CORS: 異なるオリジン（localhost:3000のReactアプリ等）からのAPIリクエストを許可
    CORS(
        app,
        origins=settings.cors_origins,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST"],
    )

    # 統一エラーハンドラ: 全エンドポイントで発生するエラーを一元的に処理
    @app.errorhandler(400)
    def bad_request(e):  # type: ignore[override]
        return (
            jsonify({"error": {"code": "BAD_REQUEST", "message": str(e)}}),
            400,
        )

    @app.errorhandler(404)
    def not_found(e):  # type: ignore[override]
        return (
            jsonify({"error": {"code": "NOT_FOUND", "message": "Not found"}}),
            404,
        )

    @app.errorhandler(Exception)
    def internal_error(e):  # type: ignore[override]
        logger.exception("Unhandled error")
        return (
            jsonify(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Internal server error",
                    }
                }
            ),
            500,
        )

    # Blueprintの登録: 各モジュールのルート・APIエンドポイントをFlaskアプリに追加
    # pages_bp: 静的ページのルーティング(/, /image, /csvTest など)
    app.register_blueprint(pages_bp)
    # text_bp: テキスト処理API(/api/split-text - LangChainによるテキスト分割)
    app.register_blueprint(text_bp)
    # iris_bp: 機械学習API(/api/iris-prediction - Iris分類予測, /api/iris-batch-prediction - バッチ予測)
    app.register_blueprint(iris_bp)
    # pdf_bp: PDF処理API(/api/extract-pdf-text - PDFからテキスト抽出)
    app.register_blueprint(pdf_bp)
    # image_bp: 画像処理API(/api/analyze-image - 画像分析)
    app.register_blueprint(image_bp)

    return app
