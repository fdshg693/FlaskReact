from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, request, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps, lru_cache
from loguru import logger

# Modern absolute imports - no sys.path manipulation needed
from llm.image import analyze_image
from llm.pdf import extract_text_from_pdf
from llm.text_splitter import split_text
from machineLearning.eval_batch import evaluate_iris_batch
from util.convert_json import convert_json_to_model_input

# Constants for file paths and configuration
APP_ROOT = Path(__file__).parent.parent
MODEL_PATH = APP_ROOT / "param" / "models_20250712_021710.pth"
SCALER_PATH = APP_ROOT / "scaler" / "scaler.joblib"

# File upload configuration
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}
MAX_IMAGE_SIZE_MB = 5
MAX_PDF_SIZE_MB = 10


@lru_cache(maxsize=1)
def get_cached_iris_prediction(features_tuple: tuple[float, ...]) -> list[str]:
    """Cache iris predictions to avoid repeated model loading."""
    features_list = [list(features_tuple)]
    return evaluate_iris_batch(features_list, MODEL_PATH, SCALER_PATH)


@lru_cache(maxsize=1)
def get_cached_batch_iris_prediction(
    features_tuple: tuple[tuple[float, ...], ...],
) -> list[str]:
    """Cache batch iris predictions to avoid repeated model loading."""
    features_list = [list(feature_tuple) for feature_tuple in features_tuple]
    return evaluate_iris_batch(features_list, MODEL_PATH, SCALER_PATH)


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
        raise ValueError("No file selected")

    if (
        "." not in file.filename
        or file.filename.rsplit(".", 1)[1].lower() not in allowed_extensions
    ):
        raise ValueError(f"File type not allowed. Allowed: {allowed_extensions}")

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > max_size_mb * 1024 * 1024:
        raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")

    return secure_filename(file.filename) or "unnamed_file"


def handle_api_errors(f):
    """Decorator to handle common API errors."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid input data"}), 400
        except FileNotFoundError as e:
            logger.error(f"File error in {f.__name__}: {e}")
            return jsonify({"error": "File processing error"}), 500
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500

    return decorated_function


# Route configuration
@app.route("/")
def serve_root_page() -> Response:
    print("Root path accessed")
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/home")
def serve_home_page() -> Response:
    # Serve static files from Flask's static folder
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/csvTest")
def serve_csv_test_page() -> Response:
    # Serve static files from Flask's static folder
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


@app.route("/sql")
def serve_sql_page() -> Response:
    if app.static_folder:
        sql_static_path = Path(app.static_folder) / "sql"
        return send_from_directory(str(sql_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


# API endpoint configuration


# Text splitting API
@app.route("/api/textSplit", methods=["POST"])
@handle_api_errors
def handle_text_split_request() -> Response:
    """
    Text splitting API endpoint
    Receives text and splits it into chunks with specified size and overlap, returns the results
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    text: str = request_data.get("text", "")
    chunk_size: int = request_data.get("chunk_size", 1000)
    chunk_overlap: int = request_data.get("chunk_overlap", 200)

    # Validate input parameters
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text content is required and must be a string"}), 400

    if not isinstance(chunk_size, int) or chunk_size < 100 or chunk_size > 5000:
        return jsonify(
            {"error": "chunk_size must be an integer between 100 and 5000"}
        ), 400

    if (
        not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        return jsonify(
            {
                "error": "chunk_overlap must be a non-negative integer less than chunk_size"
            }
        ), 400

    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return jsonify({"chunks": chunks})


# Image analysis API
@app.route("/api/image", methods=["POST"])
@handle_api_errors
def handle_image_analysis_request() -> Response:
    """
    Image analysis API endpoint
    Receives an image file, analyzes it with LLM, and returns the results
    """
    # Get image file from request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    # Validate file upload
    filename = validate_file_upload(
        image_file,
        allowed_extensions=ALLOWED_IMAGE_EXTENSIONS,
        max_size_mb=MAX_IMAGE_SIZE_MB,
    )

    logger.info(f"Processing image upload: {filename}")

    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    analysis_result = analyze_image(image_b64)

    return jsonify({"description": analysis_result})


# PDF text extraction API
@app.route("/api/pdf", methods=["POST"])
@handle_api_errors
def handle_pdf_text_extraction_request() -> Response:
    """
    PDF text extraction API endpoint
    Receives a PDF file, extracts text with LLM, and returns the results
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    pdf_file = request.files["pdf"]
    if not pdf_file:
        return jsonify({"error": "No PDF file provided"}), 400

    # Validate file upload
    filename = validate_file_upload(
        pdf_file,
        allowed_extensions=ALLOWED_PDF_EXTENSIONS,
        max_size_mb=MAX_PDF_SIZE_MB,
    )

    logger.info(f"Processing PDF upload: {filename}")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        pdf_content = pdf_file.read()
        temp_file.write(pdf_content)
        temp_file.flush()
        temp_file.seek(0)
        # Extract text
        extracted_text = extract_text_from_pdf(temp_file.name)

    return jsonify({"text": extracted_text})


# Iris dataset API
@app.route("/api/iris", methods=["POST"])
@handle_api_errors
def handle_iris_prediction_request() -> Response:
    """
    Iris species prediction API endpoint.

    Receives iris flower measurements as JSON and returns predicted species.
    Expects 4 numeric features: sepal_length, sepal_width, petal_length, petal_width.

    Returns:
        JSON response with predicted species or error message
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Convert request_data to the expected format for evaluate_iris_batch
    try:
        # Validate input is a dictionary
        if not isinstance(request_data, dict):
            return jsonify(
                {"error": "Invalid data format - expected object with numeric values"}
            ), 400

        # Extract and validate numeric values from the dictionary
        feature_values = []
        for key, value in request_data.items():
            try:
                # Validate key is string and value can be converted to float
                if not isinstance(key, str):
                    continue
                numeric_value = float(value)
                # Basic range validation for iris features
                if numeric_value < 0 or numeric_value > 20:
                    return jsonify(
                        {
                            "error": f"Invalid feature value {numeric_value}: must be between 0 and 20"
                        }
                    ), 400
                feature_values.append(numeric_value)
            except (ValueError, TypeError):
                return jsonify(
                    {"error": f"Invalid numeric value for {key}: {value}"}
                ), 400

        if len(feature_values) != 4:
            return jsonify(
                {"error": "Expected exactly 4 numeric features for iris prediction"}
            ), 400

        # Use cached prediction to avoid model reloading
        features_tuple = tuple(feature_values)
        prediction_result = get_cached_iris_prediction(features_tuple)

        logger.info(f"Processed iris prediction for features: {feature_values}")
        return jsonify({"species": prediction_result[0]})
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Data processing error in iris prediction: {e}")
        return jsonify({"error": f"Data processing error: {str(e)}"}), 400


# CSV
@app.route("/api/batch_iris", methods=["POST"])
@handle_api_errors
def handle_user_data_batch_request() -> Response:
    """
    Batch iris species prediction API endpoint.

    Receives multiple iris flower measurements as JSON array and returns predicted species for each.
    Expects array of objects with 4 numeric features each.

    Returns:
        JSON response with array of predicted species or error message
    """
    request_data = request.json
    if not request_data or "data" not in request_data:
        return jsonify({"error": "No data provided or missing 'data' field"}), 400

    try:
        iris_data_array = convert_json_to_model_input(request_data["data"])

        if not iris_data_array:
            return jsonify({"error": "No valid data rows found"}), 400

        # Use cached batch prediction to avoid model reloading
        features_tuple = tuple(tuple(row) for row in iris_data_array)
        prediction_results = get_cached_batch_iris_prediction(features_tuple)
        logger.info(f"Processed batch prediction for {len(iris_data_array)} rows")
        return jsonify({"userData": prediction_results})
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Data processing error in batch iris: {e}")
        return jsonify({"error": f"Data processing error: {str(e)}"}), 400


# Data serving
@app.route("/data/<path:path>")
def serve_data_files(path: str) -> Response:
    # Validate path to prevent directory traversal
    if ".." in path or path.startswith("/") or "\\" in path:
        return jsonify({"error": "Invalid file path"}), 400

    # Serve static files from Flask's static folder
    if app.static_folder:
        data_static_path = Path(app.static_folder) / "data"
        # Ensure the resolved path is within the data directory
        try:
            full_path = (data_static_path / path).resolve()
            if not str(full_path).startswith(str(data_static_path.resolve())):
                return jsonify({"error": "Access denied"}), 403
            return send_from_directory(str(data_static_path), path)
        except (OSError, ValueError):
            return jsonify({"error": "File not found"}), 404
    raise RuntimeError("Static folder not configured")


# SQL API Endpoints

@app.route("/api/sql/execute", methods=["POST"])
@handle_api_errors
def handle_sql_execution_request() -> Response:
    """
    SQL query execution API endpoint.
    Executes SQL queries against a sample database and returns results.
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    query = request_data.get("query", "").strip()
    database = request_data.get("database", "sample")

    if not query:
        return jsonify({"error": "SQL query is required"}), 400

    # For demo purposes, return mock data
    # In a real implementation, you would execute against a real database
    try:
        import time
        import re
        
        start_time = time.time()
        
        # Simple query parsing for demo
        query_lower = query.lower()
        
        if query_lower.startswith("select"):
            # Mock SELECT results
            if "users" in query_lower:
                mock_results = {
                    "columns": ["id", "name", "email", "created_at"],
                    "rows": [
                        {"id": 1, "name": "田中太郎", "email": "tanaka@example.com", "created_at": "2024-01-01"},
                        {"id": 2, "name": "佐藤花子", "email": "sato@example.com", "created_at": "2024-01-02"},
                        {"id": 3, "name": "鈴木一郎", "email": "suzuki@example.com", "created_at": "2024-01-03"}
                    ],
                    "rowCount": 3,
                    "executionTime": round((time.time() - start_time) * 1000, 2)
                }
            elif "products" in query_lower:
                mock_results = {
                    "columns": ["id", "name", "price", "category"],
                    "rows": [
                        {"id": 1, "name": "ノートパソコン", "price": 89800, "category": "電子機器"},
                        {"id": 2, "name": "マウス", "price": 2980, "category": "電子機器"},
                        {"id": 3, "name": "書籍", "price": 1500, "category": "本"}
                    ],
                    "rowCount": 3,
                    "executionTime": round((time.time() - start_time) * 1000, 2)
                }
            else:
                mock_results = {
                    "columns": ["result"],
                    "rows": [{"result": "クエリが正常に実行されました"}],
                    "rowCount": 1,
                    "executionTime": round((time.time() - start_time) * 1000, 2)
                }
        else:
            # Mock INSERT/UPDATE/DELETE results
            mock_results = {
                "columns": [],
                "rows": [],
                "rowCount": 0,
                "affectedRows": 1,
                "executionTime": round((time.time() - start_time) * 1000, 2)
            }

        logger.info(f"Executed SQL query: {query[:100]}...")
        return jsonify(mock_results)
        
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return jsonify({"error": f"SQL execution failed: {str(e)}"}), 500


@app.route("/api/sql/validate", methods=["POST"])
@handle_api_errors
def handle_sql_validation_request() -> Response:
    """
    SQL query validation API endpoint.
    Validates SQL syntax without executing the query.
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    query = request_data.get("query", "").strip()
    
    if not query:
        return jsonify({"valid": False, "error": "Query is empty"}), 400

    try:
        import re
        
        # Basic SQL syntax validation
        query_lower = query.lower().strip()
        
        # Check for basic SQL keywords
        sql_keywords = ["select", "insert", "update", "delete", "create", "drop", "alter"]
        starts_with_keyword = any(query_lower.startswith(keyword) for keyword in sql_keywords)
        
        if not starts_with_keyword:
            return jsonify({"valid": False, "error": "Query must start with a valid SQL keyword"})
        
        # Check for basic syntax issues
        if query_lower.count("(") != query_lower.count(")"):
            return jsonify({"valid": False, "error": "Unmatched parentheses"})
        
        # Check if query ends with semicolon (optional but good practice)
        has_semicolon = query.strip().endswith(";")
        
        return jsonify({
            "valid": True,
            "warnings": [] if has_semicolon else ["Consider adding a semicolon at the end"]
        })
        
    except Exception as e:
        logger.error(f"SQL validation error: {e}")
        return jsonify({"valid": False, "error": f"Validation failed: {str(e)}"})


@app.route("/api/sql/schema/<database>", methods=["GET"])
@handle_api_errors
def handle_schema_request(database: str) -> Response:
    """
    Database schema API endpoint.
    Returns schema information for the specified database.
    """
    # Mock schema data for demo
    mock_schema = {
        "database": database,
        "tables": [
            {
                "name": "users",
                "description": "ユーザー情報テーブル",
                "columns": [
                    {"name": "id", "type": "INTEGER", "constraints": ["PRIMARY KEY", "AUTO_INCREMENT"]},
                    {"name": "name", "type": "VARCHAR(100)", "constraints": ["NOT NULL"]},
                    {"name": "email", "type": "VARCHAR(255)", "constraints": ["UNIQUE", "NOT NULL"]},
                    {"name": "created_at", "type": "DATETIME", "constraints": ["DEFAULT CURRENT_TIMESTAMP"]}
                ],
                "sampleData": [
                    {"id": 1, "name": "田中太郎", "email": "tanaka@example.com", "created_at": "2024-01-01 10:00:00"},
                    {"id": 2, "name": "佐藤花子", "email": "sato@example.com", "created_at": "2024-01-02 11:00:00"}
                ]
            },
            {
                "name": "products",
                "description": "商品情報テーブル",
                "columns": [
                    {"name": "id", "type": "INTEGER", "constraints": ["PRIMARY KEY", "AUTO_INCREMENT"]},
                    {"name": "name", "type": "VARCHAR(200)", "constraints": ["NOT NULL"]},
                    {"name": "price", "type": "DECIMAL(10,2)", "constraints": ["NOT NULL"]},
                    {"name": "category", "type": "VARCHAR(50)", "constraints": []}
                ],
                "sampleData": [
                    {"id": 1, "name": "ノートパソコン", "price": 89800, "category": "電子機器"},
                    {"id": 2, "name": "マウス", "price": 2980, "category": "電子機器"}
                ]
            },
            {
                "name": "orders",
                "description": "注文情報テーブル",
                "columns": [
                    {"name": "id", "type": "INTEGER", "constraints": ["PRIMARY KEY", "AUTO_INCREMENT"]},
                    {"name": "user_id", "type": "INTEGER", "constraints": ["FOREIGN KEY REFERENCES users(id)"]},
                    {"name": "product_id", "type": "INTEGER", "constraints": ["FOREIGN KEY REFERENCES products(id)"]},
                    {"name": "quantity", "type": "INTEGER", "constraints": ["NOT NULL", "DEFAULT 1"]},
                    {"name": "order_date", "type": "DATETIME", "constraints": ["DEFAULT CURRENT_TIMESTAMP"]}
                ],
                "sampleData": [
                    {"id": 1, "user_id": 1, "product_id": 1, "quantity": 1, "order_date": "2024-01-03 14:30:00"},
                    {"id": 2, "user_id": 2, "product_id": 2, "quantity": 2, "order_date": "2024-01-04 09:15:00"}
                ]
            }
        ]
    }
    
    return jsonify(mock_schema)


@app.route("/api/sql/questions/<difficulty>", methods=["GET"])
@handle_api_errors  
def handle_questions_request(difficulty: str) -> Response:
    """
    SQL practice questions API endpoint.
    Returns practice questions based on difficulty level.
    """
    # Mock questions data for demo
    questions_db = {
        "beginner": [
            {
                "id": 1,
                "title": "基本的なSELECT文",
                "description": "usersテーブルから全てのユーザーの名前とメールアドレスを取得してください。",
                "sampleQuery": "SELECT name, email FROM users;",
                "expectedOutput": "name, email の列が表示される",
                "hints": ["SELECT文の基本構文を使用", "列名をカンマで区切る"]
            },
            {
                "id": 2,
                "title": "WHERE句を使った条件検索",
                "description": "productsテーブルから価格が5000円以上の商品を取得してください。",
                "sampleQuery": "SELECT * FROM products WHERE price >= 5000;",
                "expectedOutput": "価格が5000以上の商品データ",
                "hints": ["WHERE句で条件を指定", "比較演算子を使用"]
            }
        ],
        "intermediate": [
            {
                "id": 3,
                "title": "JOIN を使ったテーブル結合",
                "description": "usersテーブルとordersテーブルを結合して、ユーザー名と注文日を取得してください。",
                "sampleQuery": "SELECT u.name, o.order_date FROM users u JOIN orders o ON u.id = o.user_id;",
                "expectedOutput": "ユーザー名と注文日の組み合わせ",
                "hints": ["INNER JOINを使用", "テーブルにエイリアスを設定"]
            }
        ],
        "advanced": [
            {
                "id": 4,
                "title": "集計関数とGROUP BY",
                "description": "各ユーザーの注文回数を集計してください。",
                "sampleQuery": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name;",
                "expectedOutput": "ユーザー名と注文回数",
                "hints": ["COUNT関数を使用", "GROUP BYで集約", "LEFT JOINで全ユーザーを含める"]
            }
        ]
    }
    
    questions = questions_db.get(difficulty, questions_db["beginner"])
    
    return jsonify({
        "difficulty": difficulty,
        "questions": questions
    })


@app.route("/api/sql/ai-assist", methods=["POST"])
@handle_api_errors
def handle_ai_assist_request() -> Response:
    """
    AI SQL assistance API endpoint.
    Provides AI-powered help for SQL queries and concepts.
    """
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    question = request_data.get("question", "").strip()
    schema = request_data.get("schema", "")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Mock AI response for demo
    # In a real implementation, this would call an AI service
    mock_response = {
        "response": f"ご質問「{question}」について回答いたします。\n\nSQLクエリを書く際は以下の点に注意してください：\n1. 適切なテーブル名と列名を使用する\n2. WHERE句で条件を指定する\n3. 必要に応じてJOINを使ってテーブルを結合する",
        "suggestedQuery": "SELECT * FROM users WHERE name LIKE '%太郎%';",
        "tips": [
            "LIKE演算子でパターンマッチングが可能です",
            "ワイルドカード%を使って部分一致検索ができます",
            "インデックスが設定されている列で検索すると高速化されます"
        ],
        "resources": [
            {
                "title": "SQL基礎チュートリアル",
                "url": "#",
                "description": "SQLの基本的な構文を学習"
            }
        ]
    }
    
    return jsonify(mock_response)


if __name__ == "__main__":
    # This file should be run through the main launcher script
    # Use: python run_app.py instead
    print("⚠️  Please use 'python run_app.py' to start the application")
    print("This ensures proper Python path configuration.")
