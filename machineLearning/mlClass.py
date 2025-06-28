from pathlib import Path
import time
from typing import Tuple, List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from saveUtil import saveData2Curve, saveStudyParameter, saveData2CSV


class sNet(nn.Module):
    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class mlClass:
    def __init__(self, dataSet) -> None:
        """
        コンストラクタ
        :param dataSet: sklearnのデータセットオブジェクト
        """
        # モデル、損失関数、スケーラー、オプティマイザの初期化
        self.model = sNet()
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = StandardScaler()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        # データセットの読み込み
        self.data = dataSet.data
        self.target = dataSet.target

    def splitTrainTest(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.target,
        )

    def transform(self) -> None:
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def changeToTensorDataset(self) -> None:
        self.train_ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long),
        )
        self.test_ds = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long),
        )

    def changeToDataLoader(self) -> None:
        self.train_loader = DataLoader(self.train_ds, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=16, shuffle=False)

    def train(self, epochs: int = 20) -> Tuple[List[float], List[float]]:
        loss_list: List[float] = []
        acc_list: List[float] = []
        for _ in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                # 損失を計算
                total_loss += loss.item() * X_batch.size(0)
                # 予測ラベルと正解ラベルを比較して正解数を集計
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()

            # 平均損失を計算
            avg_loss = total_loss / len(self.train_ds)
            loss_list.append(avg_loss)
            # 平均精度を計算
            avg_acc = correct / len(self.train_ds)
            acc_list.append(avg_acc)
        return acc_list, loss_list

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                preds = self.model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        accuracy = correct / len(self.test_ds)
        return accuracy


def executeML(dataSet, epochs: int = 20):
    """
    機械学習の実行をまとめた関数
    :param dataSet: sklearnのデータセットオブジェクト
    :param epochs: 学習エポック数
    """
    machineLearning = mlClass(dataSet)
    machineLearning.splitTrainTest()
    machineLearning.transform()
    machineLearning.changeToTensorDataset()
    machineLearning.changeToDataLoader()
    acc_list, loss_list = machineLearning.train(epochs)
    return machineLearning, machineLearning.model, acc_list, loss_list


def save_model_and_curves(
    model: nn.Module, acc_list: List[float], loss_list: List[float]
) -> None:
    """
    モデルのパラメータと学習曲線を保存する関数
    :param model: 学習済みモデル
    :param acc_list: 精度のリスト
    :param loss_list: 損失のリスト
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_file = Path(__file__).resolve()

    save_param_path = current_file.parent / "../param"
    saveStudyParameter(
        model.state_dict(), str(save_param_path / f"models_{timestamp}.pth")
    )

    save_curve_path = current_file.parent / "../curveLog"
    saveData2Curve(
        loss_list,
        "loss",
        str(save_curve_path / f"loss_curve_{timestamp}.png"),
    )
    saveData2Curve(acc_list, "acc", str(save_curve_path / f"acc_curve_{timestamp}.png"))

    save_csv_path = current_file.parent / "../csvLog"
    # Convert single lists to list of lists format expected by saveData2CSV
    loss_data = [[i + 1, loss] for i, loss in enumerate(loss_list)]
    acc_data = [[i + 1, acc] for i, acc in enumerate(acc_list)]
    saveData2CSV(loss_data, str(save_csv_path / f"loss_{timestamp}.csv"))
    saveData2CSV(acc_data, str(save_csv_path / f"acc_{timestamp}.csv"))


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Irisデータセットでの実行
    iris_data = load_iris()
    iris_ml, iris_model, iris_acc_list, iris_loss_list = executeML(iris_data)
    save_model_and_curves(iris_model, iris_acc_list, iris_loss_list)
    print(f"Iris Test Accuracy: {iris_ml.evaluate():.3f}")

    # # Diabetesデータセットでの実行
    # diabetes_data = load_diabetes()
    # diabetes_ml, diabetes_model, diabetes_acc_list, diabetes_loss_list = executeML(diabetes_data)
    # save_model_and_curves(diabetes_model, diabetes_acc_list, diabetes_loss_list)
    # print(f"Diabetes Test Accuracy: {diabetes_ml.evaluate():.3f}")
