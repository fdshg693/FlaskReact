import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ★ pydanticは不要になったため、インポートを削除
from typing import List


# SimpleNetクラスの定義 (変更なし)
class SimpleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ★ 返り値の型を List[str] (文字列のリスト) に変更
def evaluateIrisBatch(input_data_list: List[List[float]]) -> List[str]:
    """
    複数のアヤメのデータ（2次元配列）を受け取り、予測されたクラス名のリストを返す関数
    """
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
        return []

    # --- 評価モードに設定 ---
    loaded_model.eval()
    print("モデルを評価モードに設定しました。")

    # --- スケーラーを読み込む ---
    try:
        scaler_dir = "scaler"
        scaler_path = os.path.join(scaler_dir, "scaler.gz")
        scaler = joblib.load(scaler_path)
        print("訓練時のスケーラーを使用します。")
    except (NameError, FileNotFoundError):
        print(f"警告: スケーラーファイル {scaler_path} が見つかりません。")
        return []

    # --- データの前処理 ---
    input_data_numpy = np.array(input_data_list, dtype=np.float32)
    print(f"\n新しいデータ (変換前):\n{input_data_numpy}")

    try:
        input_data_scaled = scaler.transform(input_data_numpy)
        print(f"新しいデータ (スケーリング後):\n{input_data_scaled}")
    except Exception as e:
        print(f"スケーリングエラー: {e}")
        return []

    # 対応するクラス名を明示 (変更なし)
    CLASS_NAMES = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    # ★ Pydanticモデルの定義は不要になったため削除

    # PyTorch Tensor に変換
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    print(f"新しいデータ (Tensor):\n{input_data_tensor}")

    # モデルで予測を実行
    with torch.no_grad():
        logits = loaded_model(input_data_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_indices = torch.argmax(probabilities, dim=1)

    # ★ 変更点: 予測されたクラス名だけをリストに格納する
    predicted_indices_list = predicted_indices.tolist()
    # リスト内包表記を使って、各インデックスを対応するクラス名に変換
    species_names = [CLASS_NAMES[i] for i in predicted_indices_list]

    # 最終的なクラス名のリストを返す
    return species_names


if __name__ == "__main__":
    # 2次元配列（リストのリスト）で複数データを定義
    input_data_list = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        # [6.7, 3.1, 4.7, 1.5],  # versicolor
        # [7.7, 3.8, 6.7, 2.2],  # virginica
        # [5.0, 3.0, 1.6, 0.2],  # setosa
    ]

    print("--- デバッグ実行開始 ---")

    # ★ 変更点: 予測を実行してクラス名のリストを`species`変数に格納
    species = evaluateIrisBatch(input_data_list)

    # ★ 変更点: 返ってきたリストをそのまま表示
    if species:
        print("\n--- 予測結果 ---")
        print(f"species = {species}")
    else:
        print("\n予測処理中にエラーが発生しました。")

    print("\n--- デバッグ実行終了 ---")
