from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class SimpleNet(nn.Module):
    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # type: ignore[no-any-return]


def evaluate_iris_batch(input_data_list: List[List[float]]) -> List[str]:
    """
    複数のアヤメのデータ（2次元配列）を受け取り、予測されたクラス名のリストを返す関数

    Args:
        input_data_list: 入力データのリスト（各要素は4つの特徴量を持つリスト）

    Returns:
        予測されたクラス名のリスト
    """
    # --- モデルを読み込む ---
    loaded_model = SimpleNet()
    param_dir = Path("param")
    model_path = param_dir / "models.pth"

    if model_path.exists():
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
    scaler_dir = Path("scaler")
    scaler_path = scaler_dir / "scaler.gz"

    try:
        scaler: StandardScaler = joblib.load(scaler_path)
        print("訓練時のスケーラーを使用します。")
    except (NameError, FileNotFoundError):
        print(f"警告: スケーラーファイル {scaler_path} が見つかりません。")
        return []

    # --- データの前処理 ---
    input_data_numpy: np.ndarray = np.array(input_data_list, dtype=np.float32)
    print(f"\n新しいデータ (変換前):\n{input_data_numpy}")

    try:
        input_data_scaled: np.ndarray = scaler.transform(input_data_numpy)
        print(f"新しいデータ (スケーリング後):\n{input_data_scaled}")
    except Exception as e:
        print(f"スケーリングエラー: {e}")
        return []

    # 対応するクラス名を明示
    CLASS_NAMES: List[str] = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    # PyTorch Tensor に変換
    input_data_tensor: torch.Tensor = torch.tensor(
        input_data_scaled, dtype=torch.float32
    )
    print(f"新しいデータ (Tensor):\n{input_data_tensor}")

    # モデルで予測を実行
    with torch.no_grad():
        logits: torch.Tensor = loaded_model(input_data_tensor)
        probabilities: torch.Tensor = F.softmax(logits, dim=1)
        predicted_indices: torch.Tensor = torch.argmax(probabilities, dim=1)

    # 予測されたクラス名だけをリストに格納する
    predicted_indices_list: List[int] = predicted_indices.tolist()
    # リスト内包表記を使って、各インデックスを対応するクラス名に変換
    species_names: List[str] = [CLASS_NAMES[i] for i in predicted_indices_list]

    # 最終的なクラス名のリストを返す
    return species_names


if __name__ == "__main__":
    # 2次元配列（リストのリスト）で複数データを定義
    input_data_list: List[List[float]] = [
        [5.1, 3.5, 1.4, 0.2],  # setosa
        # [6.7, 3.1, 4.7, 1.5],  # versicolor
        # [7.7, 3.8, 6.7, 2.2],  # virginica
        # [5.0, 3.0, 1.6, 0.2],  # setosa
    ]

    print("--- デバッグ実行開始 ---")

    # 予測を実行してクラス名のリストを`species`変数に格納
    species: List[str] = evaluate_iris_batch(input_data_list)

    # 返ってきたリストをそのまま表示
    if species:
        print("\n--- 予測結果 ---")
        print(f"species = {species}")
    else:
        print("\n予測処理中にエラーが発生しました。")

    print("\n--- デバッグ実行終了 ---")
