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
from evalBatch import evaluate_iris_batch
from util.convertJSON import jsonTo2DemensionalArray

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)  # 同一オリジン外アクセスが必要な場合のみ


# ルーティングの設定
@app.route("/")
def serve_root_page() -> Response:
    print("ルートパスにアクセスされました")
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/home")
def serve_home_page() -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/csvTest")
def serve_csv_test_page() -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        csv_test_static_path = Path(app.static_folder) / "csvTest"
        return send_from_directory(str(csv_test_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/image")
def serve_image_page() -> Response:
    if app.static_folder:
        image_static_path = Path(app.static_folder) / "image"
        return send_from_directory(str(image_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


# APIエンドポイントの設定


# テキスト分割API
@app.route("/api/textSplit", methods=["POST"])
def handle_text_split_request() -> Response | tuple[Response, int]:
    """
    テキスト分割APIエンドポイント
    テキストを受け取り、指定されたサイズとオーバーラップで分割して結果を返す
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    input_text: str = request_data.get("text", "")
    text_chunk_size: int = request_data.get("chunk_size", 1000)
    text_chunk_overlap: int = request_data.get("chunk_overlap", 200)

    text_chunks = split_text(
        input_text, chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap
    )
    return jsonify({"chunks": text_chunks})


# 画像解析API
@app.route("/api/image", methods=["POST"])
def handle_image_analysis_request() -> Response | tuple[Response, int]:
    """
    画像解析APIエンドポイント
    画像ファイルを受け取り、LLMで解析して結果を返す
    """
    # リクエストから画像ファイルを取得
    if "image" not in request.files:
        return jsonify({"error": "no file"}), 400

    uploaded_image_file = request.files["image"]
    if not uploaded_image_file:
        return jsonify({"error": "no file"}), 400

    base64_encoded_image_data = base64.b64encode(uploaded_image_file.read()).decode(
        "utf-8"
    )  # 画像データをBase64エンコード
    image_analysis_result = analyze_image(base64_encoded_image_data)  # LLMで画像を解析

    return jsonify({"description": image_analysis_result})


# PDF文字起こしAPI
@app.route("/api/pdf", methods=["POST"])
def handle_pdf_text_extraction_request() -> Response | tuple[Response, int]:
    """
    PDF文字起こしAPIエンドポイント
    PDFファイルを受け取り、LLMで文字起こしして結果を返す
    """
    if "pdf" not in request.files:
        return jsonify({"error": "no file"}), 400

    uploaded_pdf_file = request.files["pdf"]
    if not uploaded_pdf_file:
        return jsonify({"error": "no file"}), 400

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temporary_pdf_file:
        pdf_file_content = uploaded_pdf_file.read()
        temporary_pdf_file.write(pdf_file_content)
        temporary_pdf_file.flush()
        temporary_pdf_file.seek(0)
        # LLMでPDFを解析
        # テキスト抽出
        extracted_text_content = extract_text_from_pdf(temporary_pdf_file.name)

    return jsonify({"text": extracted_text_content})


# irisデータセット用API
@app.route("/api/iris", methods=["POST"])
def handle_iris_prediction_request():
    iris_input_data = request.json
    if not iris_input_data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Convert iris_input_data to the expected format for evaluate_iris_batch
    try:
        # Assuming iris_input_data is a dict with numeric values that need to be converted to a list
        if isinstance(iris_input_data, dict):
            # Extract numeric values from the dictionary
            iris_feature_values = [
                float(value)
                for value in iris_input_data.values()
                if isinstance(value, (int, float, str))
                and str(value).replace(".", "").isdigit()
            ]
            prediction_result = evaluate_iris_batch([iris_feature_values])
        else:
            return jsonify({"error": "Invalid data format"}), 400

        print(f"Received iris data: {iris_input_data}")
        return jsonify({"species": prediction_result[0]})
    except (ValueError, TypeError, IndexError) as e:
        return jsonify({"error": f"Data processing error: {str(e)}"}), 400


# CSV
@app.route("/api/userData", methods=["POST"])
def handle_user_data_batch_request():
    user_submitted_data = request.json
    if not user_submitted_data or "data" not in user_submitted_data:
        return jsonify({"error": "No data provided or missing 'data' field"}), 400

    try:
        converted_iris_data_array = jsonTo2DemensionalArray(user_submitted_data["data"])
        batch_prediction_results = evaluate_iris_batch(converted_iris_data_array)
        return jsonify({"userData": batch_prediction_results})
    except (ValueError, TypeError, KeyError) as e:
        return jsonify({"error": f"Data processing error: {str(e)}"}), 400


# データ配信用
@app.route("/data/<path:path>")
def serve_data_files(path: str) -> Response:
    # Flaskのstaticフォルダから静的ファイルを提供
    if app.static_folder:
        data_static_path = Path(app.static_folder) / "data"
        return send_from_directory(str(data_static_path), path)
    raise RuntimeError("Static folder not configured")


if __name__ == "__main__":
    # ポートやデバッグモードはお好みで調整
    app.run(host="0.0.0.0", port=8000, debug=True)
