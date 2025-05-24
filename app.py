from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import math
import random
import base64

from llm.image import AnalyzeImage

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # 同一オリジン外アクセスが必要な場合のみ


# ルーティングの設定
@app.route("/")
def root():
    return send_from_directory(app.static_folder + "/home", "index.html")


@app.route("/home")
def home():
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/home", "index.html")


@app.route("/csvTest")
def static_proxy():
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/csvTest", "index.html")


@app.route("/image")
def image():
    return send_from_directory(app.static_folder + "/image", "index.html")


# APIエンドポイントの設定


# 画像解析API
@app.route("/api/image", methods=["POST"])
def apiImage():
    """
    画像解析APIエンドポイント
    画像ファイルを受け取り、LLMで解析して結果を返す
    """
    # リクエストから画像ファイルを取得
    if "image" not in request.files:
        return {"error": "no file"}, 400
    f = request.files.get("image")
    image_data = base64.b64encode(f.read()).decode(
        "utf-8"
    )  # 画像データをBase64エンコード
    result = AnalyzeImage(image_data)  # LLMで画像を解析

    return jsonify({"description": result})


@app.route("/api/iris", methods=["POST"])
def iris():
    irisSpecies = ["setosa", "versicolor", "virginica"]
    randomSpecies = irisSpecies[math.floor(random.random() * len(irisSpecies))]
    return jsonify({"species": randomSpecies})


@app.route("/api/userData", methods=["POST"])
def userData():
    userData = {"name": "太郎", "age": 30}
    return jsonify({"userData": userData})


# データ配信用
@app.route("/data/<path:path>")
def data_proxy(path):
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/data", path)


if __name__ == "__main__":
    # ポートやデバッグモードはお好みで調整
    app.run(host="0.0.0.0", port=8000, debug=True)
