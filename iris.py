from pathlib import Path
from typing import Tuple, List, Optional, Union
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


class SimpleNet(nn.Module):
    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MachineLearning:
    model: SimpleNet = SimpleNet()
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    scaler: StandardScaler = StandardScaler()
    optimizer: torch.optim.SGD = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9
    )

    def __init__(
        self,
        dataSet: Bunch,
        epochs: int = 20,
        param_dir: Union[str, Path] = "param",
        log_dir: Union[str, Path] = "log",
        scaler_dir: Union[str, Path] = "scaler",
    ) -> None:
        self.data: np.ndarray = dataSet.data
        self.target: np.ndarray = dataSet.target
        self.epochs: int = epochs
        self.param_dir: Path = Path(param_dir)
        self.log_dir: Path = Path(log_dir)
        self.scaler_dir: Path = Path(scaler_dir)

        # ディレクトリ作成
        self.param_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.scaler_dir.mkdir(exist_ok=True)

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

    def train(self) -> None:
        self.loss_list: List[float] = []
        self.acc_list: List[float] = []
        train_dataset_size = len(self.train_ds)

        for epoch in range(1, self.epochs + 1):
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
            avg_loss = total_loss / train_dataset_size
            self.loss_list.append(avg_loss)
            # 平均精度を計算
            avg_acc = correct / train_dataset_size
            self.acc_list.append(avg_acc)
            # ログ出力
            print(f"Epoch{epoch:2d} Loss:{avg_loss:.4f}, Acc:{avg_acc:4f}")

    def saveStudyParameter(self) -> None:
        save_path = self.param_dir / "models.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"モデルの state_dict を {save_path} に保存しました。")

    def saveScaler(self) -> None:
        """
        学習済みのスケーラーをjoblibを使って保存する。
        """
        scaler_path = self.scaler_dir / "scaler.gz"
        joblib.dump(self.scaler, scaler_path)
        print(f"スケーラーを {scaler_path} に保存しました。")

    def saveTrainCurve(self, data: List[float], data_name: str) -> None:
        file_path = self.log_dir / f"training_{data_name}_curve.png"
        plt.figure()
        plt.plot(range(1, self.epochs + 1), data)
        plt.xlabel("Epoch")
        plt.ylabel(f"Average {data_name}")
        plt.title(f"Training {data_name} Curve")
        plt.savefig(file_path)
        plt.close()
        print(f"{file_path} を保存しました。")

    def evaluate(self) -> None:
        self.model.eval()
        correct = 0
        test_dataset_size = len(self.test_ds)

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                preds = self.model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        accuracy = correct / test_dataset_size
        print(f"Test Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    iris_data: Bunch = load_iris()  # type: ignore
    machineLearning = MachineLearning(iris_data, 20)
    machineLearning.splitTrainTest()
    machineLearning.transform()
    machineLearning.changeToTensorDataset()
    machineLearning.changeToDataLoader()
    machineLearning.train()
    machineLearning.saveStudyParameter()
    machineLearning.saveScaler()
    machineLearning.saveTrainCurve(machineLearning.loss_list, "loss")
    machineLearning.saveTrainCurve(machineLearning.acc_list, "acc")
    machineLearning.evaluate()
