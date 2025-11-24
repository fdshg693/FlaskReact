from __future__ import annotations

from flask import Blueprint, Response, send_from_directory, jsonify

from config.paths import PATHS

# ページルーティング用のBlueprint定義
# 静的HTMLページの配信を担当
pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def serve_root_page() -> Response:
    """
    ルートパス ('/') へのアクセス時にホームページを返す

    Returns:
        Response: static/home/index.html を配信
    """
    home_static_path = PATHS.flask_static / "home"
    return send_from_directory(str(home_static_path), "index.html")


@pages_bp.route("/home")
def serve_home_page() -> Response:
    """
    '/home' パスへのアクセス時にホームページを返す
    Iris予測(単一・バッチ)機能を提供するページ

    Returns:
        Response: static/home/index.html を配信
    """
    home_static_path = PATHS.flask_static / "home"
    return send_from_directory(str(home_static_path), "index.html")


@pages_bp.route("/csvTest")
def serve_csv_test_page() -> Response:
    """
    '/csvTest' パスへのアクセス時にCSVテストページを返す
    CSVファイルのアップロードと処理をテストするページ

    Returns:
        Response: static/csvTest/index.html を配信
    """
    csv_test_static_path = PATHS.flask_static / "csvTest"
    return send_from_directory(str(csv_test_static_path), "index.html")


@pages_bp.route("/image")
def serve_image_page() -> Response:
    """
    '/image' パスへのアクセス時に画像処理ページを返す
    画像分析・PDF処理・テキスト分割などのLLM機能を提供するページ

    Returns:
        Response: static/image/index.html を配信
    """
    image_static_path = PATHS.flask_static / "image"
    return send_from_directory(str(image_static_path), "index.html")


@pages_bp.route("/data/<path:path>")
def serve_data_files(path: str) -> Response:
    """
    '/data/<path>' パスへのアクセス時にstatic/data配下のファイルを配信

    セキュリティ対策:
    - ディレクトリトラバーサル攻撃を防止
    - パス検証により不正なアクセスをブロック
    - data ディレクトリ外へのアクセスを禁止

    Args:
        path: 要求されたファイルパス (例: 'sample.csv')

    Returns:
        Response: 要求されたファイル、またはエラーレスポンス
            - 400: 不正なパス形式
            - 403: アクセス権限なし
            - 404: ファイルが見つからない
    """
    # パス検証: ディレクトリトラバーサル攻撃を防止
    # ".."、絶対パス、バックスラッシュを含むパスを拒否
    if ".." in path or path.startswith("/") or "\\" in path:
        return jsonify(
            {"error": {"code": "INVALID_PATH", "message": "Invalid file path"}}
        ), 400

    data_static_path = PATHS.flask_static / "data"
    try:
        # パスを正規化してdataディレクトリ内に収まっているか検証
        full_path = (data_static_path / path).resolve()
        if not str(full_path).startswith(str(data_static_path.resolve())):
            return jsonify(
                {"error": {"code": "ACCESS_DENIED", "message": "Access denied"}}
            ), 403
        return send_from_directory(str(data_static_path), path)
    except (OSError, ValueError):
        # ファイルが存在しない、または読み取りエラー
        return jsonify(
            {"error": {"code": "NOT_FOUND", "message": "File not found"}}
        ), 404
