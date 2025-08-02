from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, request, Response, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps
from loguru import logger

# Modern absolute imports - no sys.path manipulation needed
from llm.image import analyze_image
from llm.pdf import extract_text_from_pdf
from llm.text_splitter import split_text
from machineLearning.eval_batch import evaluate_iris_batch
from util.convert_json import convert_json_to_model_input

app = Flask(__name__, static_folder="../static", static_url_path="")
# Secure CORS configuration - only allow specific origins
CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST"],
)


def validate_file_upload(
    file, allowed_extensions: set[str], max_size_mb: int = 10
) -> str:
    """Validate uploaded file for security and size constraints."""
    if not file or file.filename == "":
        abort(400, "No file selected")

    if (
        "." not in file.filename
        or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions
    ):
        abort(400, f"File type not allowed. Allowed: {allowed_extensions}")

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > max_size_mb * 1024 * 1024:
        abort(400, f"File too large. Maximum size: {max_size_mb}MB")

    return secure_filename(file.filename) or "unnamed_file"


def handle_api_errors(f):
    """Decorator to handle common API errors."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid input data", "details": str(e)}), 400
        except FileNotFoundError as e:
            logger.error(f"File error in {f.__name__}: {e}")
            return jsonify({"error": "File processing error"}), 500
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return decorated_function


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
@handle_api_errors
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

    if not input_text.strip():
        return jsonify({"error": "Text content is required"}), 400

    text_chunks = split_text(
        input_text, chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap
    )
    return jsonify({"chunks": text_chunks})


# 画像解析API
@app.route("/api/image", methods=["POST"])
@handle_api_errors
def handle_image_analysis_request() -> Response | tuple[Response, int]:
    """
    画像解析APIエンドポイント
    画像ファイルを受け取り、LLMで解析して結果を返す
    """
    # リクエストから画像ファイルを取得
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    uploaded_image_file = request.files["image"]
    if not uploaded_image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Validate file upload
    filename = validate_file_upload(
        uploaded_image_file,
        allowed_extensions={"png", "jpg", "jpeg", "gif"},
        max_size_mb=5,
    )

    logger.info(f"Processing image upload: {filename}")

    base64_encoded_image_data = base64.b64encode(uploaded_image_file.read()).decode(
        "utf-8"
    )  # 画像データをBase64エンコード
    image_analysis_result = analyze_image(base64_encoded_image_data)  # LLMで画像を解析

    return jsonify({"description": image_analysis_result})


# PDF文字起こしAPI
@app.route("/api/pdf", methods=["POST"])
@handle_api_errors
def handle_pdf_text_extraction_request() -> Response | tuple[Response, int]:
    """
    PDF文字起こしAPIエンドポイント
    PDFファイルを受け取り、LLMで文字起こしして結果を返す
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    uploaded_pdf_file = request.files["pdf"]
    if not uploaded_pdf_file:
        return jsonify({"error": "No PDF file provided"}), 400

    # Validate file upload
    filename = validate_file_upload(
        uploaded_pdf_file, allowed_extensions={"pdf"}, max_size_mb=10
    )

    logger.info(f"Processing PDF upload: {filename}")

    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temporary_pdf_file:
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
@handle_api_errors
def handle_iris_prediction_request() -> Response | tuple[Response, int]:
    iris_input_data = request.json
    if not iris_input_data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Convert iris_input_data to the expected format for evaluate_iris_batch
    try:
        # Assuming iris_input_data is a dict with numeric values that need to be converted to a list
        if isinstance(iris_input_data, dict):
            # Extract numeric values from the dictionary using robust float conversion
            iris_feature_values = []
            for value in iris_input_data.values():
                try:
                    # Use try-except for robust numeric validation
                    numeric_value = float(value)
                    iris_feature_values.append(numeric_value)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

            if len(iris_feature_values) != 4:
                return jsonify(
                    {"error": "Expected 4 numeric features for iris prediction"}
                ), 400

            # Define paths for model and scaler
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "param" / "models_20250712_021710.pth"
            scaler_path = current_dir / "scaler" / "scaler.joblib"

            prediction_result = evaluate_iris_batch(
                [iris_feature_values], model_path, scaler_path
            )
        else:
            return jsonify(
                {"error": "Invalid data format - expected object with numeric values"}
            ), 400

        logger.info(f"Received iris data: {iris_input_data}")
        return jsonify({"species": prediction_result[0]})
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Data processing error in iris prediction: {e}")
        return jsonify({"error": f"Data processing error: {str(e)}"}), 400


# CSV
@app.route("/api/batch_iris", methods=["POST"])
@handle_api_errors
def handle_user_data_batch_request() -> Response | tuple[Response, int]:
    user_submitted_data = request.json
    if not user_submitted_data or "data" not in user_submitted_data:
        return jsonify({"error": "No data provided or missing 'data' field"}), 400

    try:
        converted_iris_data_array = convert_json_to_model_input(
            user_submitted_data["data"]
        )

        if not converted_iris_data_array:
            return jsonify({"error": "No valid data rows found"}), 400

        # Define paths for model and scaler
        current_dir = Path(__file__).parent.parent
        model_path = current_dir / "param" / "models_20250712_021710.pth"
        scaler_path = current_dir / "scaler" / "scaler.joblib"

        batch_prediction_results = evaluate_iris_batch(
            converted_iris_data_array, model_path, scaler_path
        )
        logger.info(
            f"Processed batch prediction for {len(converted_iris_data_array)} rows"
        )
        return jsonify({"userData": batch_prediction_results})
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Data processing error in batch iris: {e}")
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
    # This file should be run through the main launcher script
    # Use: python run_app.py instead
    print("⚠️  Please use 'python run_app.py' to start the application")
    print("This ensures proper Python path configuration.")
