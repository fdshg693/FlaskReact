#!/usr/bin/env python3
"""Simple Flask server for testing login functionality."""

from flask import Flask, render_template_string, send_from_directory
from flask_cors import CORS
from pathlib import Path

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(
    app,
    origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST"],
)

@app.route("/")
def serve_root_page():
    print("ルートパスにアクセスされました")
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")

@app.route("/home")
def serve_home_page():
    if app.static_folder:
        home_static_path = Path(app.static_folder) / "home"
        return send_from_directory(str(home_static_path), "index.html")
    raise RuntimeError("Static folder not configured")

if __name__ == "__main__":
    print("🚀 Starting simple FlaskReact application...")
    app.run(host="0.0.0.0", port=8000, debug=True)