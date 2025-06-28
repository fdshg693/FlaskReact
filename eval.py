import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pydantic
from pydantic import BaseModel
from typing import List


# SimpleNetクラスの定義
class SimpleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def evaluateIris(inputData):
    # --- モデルを読み込む ---
    loaded_model = SimpleNet()
    param_dir = "param"
    model_path = os.path.join(param_dir, "models.pth")

    if os.path.exists(model_path):
        loaded_model.load_state_dict(torch.load(model_path))
        print(f"モデルのパラメータを {model_path} から読み込みました。")
    else:
        print(
            f"エラー: パス {model_path} にモデルファイルが見つかりません。処理を中断します。"
        )
        exit()  # モデルがない場合は終了

    # --- 評価モードに設定 ---
    loaded_model.eval()
    print("モデルを評価モードに設定しました。")

    try:
        # machineLearning は最初のスクリプトで定義されたインスタンス名と仮定
        scaler_dir = "scaler"
        scaler_path = os.path.join(scaler_dir, "scaler.gz")
        scaler = joblib.load(scaler_path)
        print("訓練時のスケーラーを使用します。")
    except NameError:
        print("警告: 訓練時の 'machineLearning.scaler' が見つかりません。")
        print(
            "予測のためには、訓練に使用したのと同じスケーラーで新しいデータを変換する必要があります。"
        )
        print("ダミーのスケーラーを作成しますが、これは正しい結果を保証しません。")
        # 実際のアプリケーションでは、ここでエラーにするか、保存されたスケーラーをロードする処理を実装してください。
        scaler = StandardScaler()

    inputData_numpy = np.array([inputData], dtype=np.float32)
    print(f"\n新しいデータ (変換前):\n{inputData_numpy}")

    # 新しいデータの前処理 (スケーリング)
    try:
        inputData_scaled = scaler.transform(inputData_numpy)
        print(f"新しいデータ (スケーリング後):\n{inputData_scaled}")
    except AttributeError:
        print(
            "エラー: スケーラーが正しく初期化されていないか、'transform' メソッドがありません。"
        )
        print("訓練時のスケーラーを正しくロードまたは参照しているか確認してください。")
        exit()
    except Exception as e:  # NotFittedErrorなど sklearn.exceptions.NotFittedError
        print(f"スケーリングエラー: {e}")
        print("スケーラーが訓練データで 'fit' されていない可能性があります。")
        print("訓練時のスケーラーを正しくロードまたは参照しているか確認してください。")
        exit()

    # 対応するクラス名を明示
    CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    class IrisInput(BaseModel):
        features: List[float]  # [5.1, 3.5, 1.4, 0.2] のような4つの数値のリスト

    # 3. レスポンスデータの型定義 (任意ですが推奨)
    class PredictionResult(BaseModel):
        class_name: str
        probability: float

    class PredictionResponse(BaseModel):
        predicted_class: str
        probabilities: List[PredictionResult]

    # 4. PyTorch Tensor に変換
    inputData_tensor = torch.tensor(inputData_scaled, dtype=torch.float32)
    print(f"新しいデータ (Tensor):\n{inputData_tensor}")

    # 5. モデルで予測を実行
    with torch.no_grad():  # 勾配計算をオフにして、メモリ効率を上げ、計算を高速化
        logits = loaded_model(inputData_tensor)
        # ★ ロジットを確率に変換 (Softmax関数)
        probabilities = F.softmax(logits, dim=1)

    # torch.tensor型からpythonのlist型に変換する
    """
    probabilitiesは [[0.9, 0.05, 0.05]] のような形なので、[0]で中身を取り出す
    ※今回は1つだけの入力のため、probabilities[0]が[0.9, 0.05, 0.05]を表す
    """
    probs_list = probabilities[0].tolist()

    # 予測されたクラスのインデックスを取得（出力を決定している場所）
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class_name = CLASS_NAMES[predicted_index]

    # 各クラスの確率を整形
    """
    以下のような形に整形
    [
        { "class_name": "Iris setosa", "probability": 0.05 },
        { "class_name": "Iris versicolor", "probability": 0.90 },
        { "class_name": "Iris virginica", "probability": 0.05 }
    ]
    """
    prob_results = []
    for i, prob in enumerate(probs_list):
        prob_results.append(
            PredictionResult(class_name=CLASS_NAMES[i], probability=prob)
        )

    # 最終的なレスポンスを返す
    return PredictionResponse(
        predicted_class=predicted_class_name, probabilities=prob_results
    )


if __name__ == "__main__":
    # デバッグ用のサンプル入力データ
    inputData = [5.1, 3.5, 1.4, 0.2]

    print("--- デバッグ実行開始 ---")

    # 予測を実行して結果を取得
    prediction = evaluateIris(inputData)

    # 結果をコンソールに表示
    # (Pydanticモデルはそのままprintできます)
    print(prediction)

    # もし、より綺麗なJSON形式で見たければ以下のようにします
    # import json
    # print(json.dumps(prediction.model_dump(), indent=2, ensure_ascii=False))

    print("--- デバッグ実行終了 ---")
