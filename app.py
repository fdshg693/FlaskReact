from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import math
import random


app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # 同一オリジン外アクセスが必要な場合のみ


@app.route("/")
def root():
    return send_from_directory(app.static_folder + "/home", "index.html")


@app.route("/home")
def home():
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/home", "index.html")


@app.route("/api/image")
def image():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    randomLetter = letters[math.floor(random.random() * len(letters))]
    return jsonify({"letter": randomLetter})


@app.route("/api/iris", methods=["POST"])
def iris():
    irisSpecies = ["setosa", "versicolor", "virginica"]
    randomSpecies = irisSpecies[math.floor(random.random() * len(irisSpecies))]
    return jsonify({"species": randomSpecies})


@app.route("/csvTest")
def static_proxy():
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/csvTest", "index.html")


@app.route("/api/userData", methods=["POST"])
def userData():
    userData = {"name": "太郎", "age": 30}
    return jsonify({"userData": "finished"})


# データ配信用
@app.route("/data/<path:path>")
def data_proxy(path):
    # Flaskのstaticフォルダから静的ファイルを提供
    return send_from_directory(app.static_folder + "/data", path)


if __name__ == "__main__":
    # ポートやデバッグモードはお好みで調整
    app.run(host="0.0.0.0", port=8000, debug=True)
