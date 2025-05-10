from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import math
import random


app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # 同一オリジン外アクセスが必要な場合のみ

@app.route("/api/message")
def message():
    return jsonify({"text": "こんにちは、React＋Flask アプリです！"})

@app.route("/api/image")
def image():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    randomLetter = letters[math.floor(random.random() * len(letters))]
    return jsonify({"text": randomLetter})

@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    # ポートやデバッグモードはお好みで調整
    app.run(host="0.0.0.0", port=5000, debug=True)
