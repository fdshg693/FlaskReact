from __future__ import annotations

import base64
import random
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, request, Response
from flask_cors import CORS

# Add parent directory to path for imports (must be done before local imports)
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.image import analyze_image
from llm.pdf import extract_text_from_pdf
from llm.textSplit import split_text

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)  # 同一オリジン外アクセスが必要な場合のみ


# ルーティングの設定
@app.route("/")
def root() -> Response:
    print("ルートパスにアクセスされました")
    if app.static_folder:
        static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/home")
def home() -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/csvTest")
def static_proxy() -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        static_path = Path(app.static_folder) / "csvTest"
        return send_from_directory(str(static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/image")
def image() -> Response:
    if app.static_folder:
        static_path = Path(app.static_folder) / "image"
        return send_from_directory(str(static_path), "index.html")
    raise RuntimeError("Static folder not configured")


# APIエンドポイントの設定


# テキスト分割API
@app.route("/api/textSplit", methods=["POST"])
def api_text_split() -> Response | tuple[Response, int]:
    """
    テキスト分割APIエンドポイント
    テキストを受け取り、指定されたサイズとオーバーラップで分割して結果を返す
    """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    text: str = data.get("text", "")
    chunk_size: int = data.get("chunk_size", 1000)
    chunk_overlap: int = data.get("chunk_overlap", 200)

    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return jsonify({"chunks": chunks})


# 画像解析API
@app.route("/api/image", methods=["POST"])
def api_image() -> Response | tuple[Response, int]:
    """
    画像解析APIエンドポイント
    画像ファイルを受け取り、LLMで解析して結果を返す
    """
    # リクエストから画像ファイルを取得
    if "image" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["image"]
    if not file:
        return jsonify({"error": "no file"}), 400

    image_data = base64.b64encode(file.read()).decode(
        "utf-8"
    )  # 画像データをBase64エンコード
    result = analyze_image(image_data)  # LLMで画像を解析

    return jsonify({"description": result})


# PDF文字起こしAPI
@app.route("/api/pdf", methods=["POST"])
def api_pdf() -> Response | tuple[Response, int]:
    """
    PDF文字起こしAPIエンドポイント
    PDFファイルを受け取り、LLMで文字起こしして結果を返す
    """
    if "pdf" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["pdf"]
    if not file:
        return jsonify({"error": "no file"}), 400

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        content = file.read()
        tmp.write(content)
        tmp.flush()
        tmp.seek(0)
        # LLMでPDFを解析
        # テキスト抽出
        raw_text = extract_text_from_pdf(tmp.name)

    return jsonify({"text": raw_text})


@app.route("/api/iris", methods=["POST"])
def iris() -> Response:
    iris_species = ["setosa", "versicolor", "virginica"]
    random_species = random.choice(iris_species)
    return jsonify({"species": random_species})


@app.route("/api/userData", methods=["POST"])
def user_data() -> Response:
    user_data_dict = {"name": "太郎", "age": 30}
    return jsonify({"userData": user_data_dict})


# データ配信用
@app.route("/data/<path:path>")
def data_proxy(path: str) -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        static_path = Path(app.static_folder) / "data"
        return send_from_directory(str(static_path), path)
    raise RuntimeError("Static folder not configured")


if __name__ == "__main__":
    # ポートやデバッグモードはお好みで調整
    app.run(host="0.0.0.0", port=8000, debug=True)
