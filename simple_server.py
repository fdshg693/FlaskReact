#!/usr/bin/env python3
"""Simple Flask server with login functionality for testing."""

from flask import Flask, send_from_directory, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from pathlib import Path

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = "dev-secret-key-change-in-production"

CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST"],
    supports_credentials=True,
)

# Simple in-memory user store for demo purposes
USERS = {
    "admin": {
        "password_hash": generate_password_hash("password123"),
        "username": "admin"
    },
    "user": {
        "password_hash": generate_password_hash("user123"),
        "username": "user"
    }
}

def login_required(f):
    """Decorator to require login for certain routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def serve_root_page():
    print("ルートパスにアクセスされました")
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")

@app.route("/login-demo")
def serve_login_demo():
    """Serve the login demo page."""
    if app.static_folder:
        return send_from_directory(app.static_folder, "login_demo.html")
    raise RuntimeError("Static folder not configured")

@app.route("/home")
def serve_home_page():
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")

# Authentication endpoints
@app.route("/api/login", methods=["POST"])
def handle_login_request():
    """Handle user login requests."""
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    username = request_data.get("username", "")
    password = request_data.get("password", "")
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    user = USERS.get(username)
    if user and check_password_hash(user["password_hash"], password):
        session["user_id"] = username
        print(f"User {username} logged in successfully")
        return jsonify({
            "message": "Login successful",
            "user": {"username": username}
        })
    else:
        print(f"Failed login attempt for username: {username}")
        return jsonify({"error": "Invalid username or password"}), 401

@app.route("/api/logout", methods=["POST"])
def handle_logout_request():
    """Handle user logout requests."""
    user_id = session.get("user_id")
    session.pop("user_id", None)
    if user_id:
        print(f"User {user_id} logged out")
    return jsonify({"message": "Logout successful"})

@app.route("/api/auth/status", methods=["GET"])
def check_auth_status():
    """Check if user is currently authenticated."""
    if "user_id" in session:
        return jsonify({
            "authenticated": True,
            "user": {"username": session["user_id"]}
        })
    else:
        return jsonify({"authenticated": False})

# Mock iris API endpoint for testing (login required)
@app.route("/api/iris", methods=["POST"])
@login_required
def handle_iris_prediction_request():
    """Mock iris prediction endpoint."""
    request_data = request.json
    if not request_data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Mock response
    return jsonify({"species": "Mock Iris Species - Authentication Working!"})

if __name__ == "__main__":
    print("🚀 Starting simple FlaskReact application with login...")
    app.run(host="0.0.0.0", port=8000, debug=True)