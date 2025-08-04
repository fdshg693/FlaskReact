#!/usr/bin/env python3
"""Simplified Flask server for SQL interface demonstration."""

from __future__ import annotations

import json
import time
import re
from pathlib import Path

from flask import Flask, jsonify, send_from_directory, request, Response
from flask_cors import CORS
from loguru import logger

# Simple Flask app for demonstration
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(
    app,
    origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST"],
)


def handle_api_errors(f):
    """Decorator to handle common API errors."""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": "Invalid input data"}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function


# Route configuration
@app.route("/")
def serve_root_page() -> Response:
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/home")
def serve_home_page() -> Response:
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/sql")
def serve_sql_page() -> Response:
    if app.static_folder:
        sql_static_path = Path(app.static_folder) / "sql"
        return send_from_directory(str(sql_static_path), "index.html")
    raise RuntimeError("Static folder not configured")


@app.route("/sql/demo")
def serve_sql_demo_page() -> Response:
    if app.static_folder:
        sql_static_path = Path(app.static_folder) / "sql"
        return send_from_directory(str(sql_static_path), "demo.html")
    raise RuntimeError("Static folder not configured")


# SQL API Endpoints

@app.route("/api/sql/execute", methods=["POST"])
@handle_api_errors
def handle_sql_execution_request() -> Response:
    """Execute SQL query against the database."""
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    query = request_data.get("query", "").strip()
    database = request_data.get("database", "sample")

    if not query:
        return jsonify({"error": "SQL query is required"}), 400

    try:
        start_time = time.time()
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
    """Validate SQL query syntax."""
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    query = request_data.get("query", "").strip()
    
    if not query:
        return jsonify({"valid": False, "error": "Query is empty"}), 400

    try:
        query_lower = query.lower().strip()
        sql_keywords = ["select", "insert", "update", "delete", "create", "drop", "alter"]
        starts_with_keyword = any(query_lower.startswith(keyword) for keyword in sql_keywords)
        
        if not starts_with_keyword:
            return jsonify({"valid": False, "error": "Query must start with a valid SQL keyword"})
        
        if query_lower.count("(") != query_lower.count(")"):
            return jsonify({"valid": False, "error": "Unmatched parentheses"})
        
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
    """Get database schema information."""
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
            }
        ]
    }
    
    return jsonify(mock_schema)


@app.route("/api/sql/questions/<difficulty>", methods=["GET"])
@handle_api_errors  
def handle_questions_request(difficulty: str) -> Response:
    """Get SQL practice questions."""
    questions_db = {
        "beginner": [
            {
                "id": 1,
                "title": "基本的なSELECT文",
                "description": "usersテーブルから全てのユーザーの名前とメールアドレスを取得してください。",
                "sampleQuery": "SELECT name, email FROM users;",
                "expectedOutput": "name, email の列が表示される",
                "hints": ["SELECT文の基本構文を使用", "列名をカンマで区切る"]
            }
        ],
        "intermediate": [
            {
                "id": 2,
                "title": "WHERE句を使った条件検索",
                "description": "productsテーブルから価格が5000円以上の商品を取得してください。",
                "sampleQuery": "SELECT * FROM products WHERE price >= 5000;",
                "expectedOutput": "価格が5000以上の商品データ",
                "hints": ["WHERE句で条件を指定", "比較演算子を使用"]
            }
        ],
        "advanced": [
            {
                "id": 3,
                "title": "集計関数とGROUP BY",
                "description": "各カテゴリの商品数を集計してください。",
                "sampleQuery": "SELECT category, COUNT(*) as count FROM products GROUP BY category;",
                "expectedOutput": "カテゴリ名と商品数",
                "hints": ["COUNT関数を使用", "GROUP BYで集約"]
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
    """AI SQL assistance endpoint."""
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400

    question = request_data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Mock AI response
    mock_response = {
        "response": f"ご質問「{question}」について回答いたします。\n\nSQLクエリを書く際は以下の点に注意してください：\n1. 適切なテーブル名と列名を使用する\n2. WHERE句で条件を指定する\n3. 必要に応じてJOINを使ってテーブルを結合する",
        "suggestedQuery": "SELECT * FROM users WHERE name LIKE '%太郎%';",
        "tips": [
            "LIKE演算子でパターンマッチングが可能です",
            "ワイルドカード%を使って部分一致検索ができます"
        ]
    }
    
    return jsonify(mock_response)


if __name__ == "__main__":
    logger.info("Starting SQL interface server on port 8000")
    app.run(host="127.0.0.1", port=8000, debug=True)