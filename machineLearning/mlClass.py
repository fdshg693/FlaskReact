import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from saveUtil import saveData2Curve, saveStudyParameter


class sNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class mlClass:
    model = sNet()
    criterion = nn.CrossEntropyLoss()
    scaler = StandardScaler()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def __init__(self, dataSet, epochs=20):
        """
        コンストラクタ
        :param dataSet: sklearnのデータセットオブジェクト
        :param epochs: 学習エポック数
        """
        self.data = dataSet.data
        self.target = dataSet.target
        self.epochs = epochs

    def splitTrainTest(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.target,
        )

    def transform(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def changeToTensorDataset(self):
        self.train_ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.long),
        )
        self.test_ds = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long),
        )

    def changeToDataLoader(self):
        self.train_loader = DataLoader(self.train_ds, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=16, shuffle=False)

    def train(self):
        self.loss_list = []
        self.acc_list = []
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
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.loss_list.append(avg_loss)
            # 平均精度を計算
            avg_acc = correct / len(self.train_loader.dataset)
            self.acc_list.append(avg_acc)
            # ログ出力
            print(f"Epoch{epoch:2d} Loss:{avg_loss:.4f}, Acc:{avg_acc:4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                preds = self.model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        accuracy = correct / len(self.test_loader.dataset)
        print(f"Test Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # Irisデータセットを使用してmlClassをテスト
    from sklearn.datasets import load_iris

    machineLearning = mlClass(load_iris(), 20)
    machineLearning.splitTrainTest()
    machineLearning.transform()
    machineLearning.changeToTensorDataset()
    machineLearning.changeToDataLoader()
    # 学習
    machineLearning.train()
    # モデルのパラメータを保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_param_path = os.path.join(os.path.dirname(__file__), "../param")
    saveStudyParameter(
        machineLearning.model.state_dict(), f"{save_param_path}/models_{timestamp}.pth"
    )
    # 学習曲線を保存
    save_curve_path = os.path.join(os.path.dirname(__file__), "../curveLog")
    saveData2Curve(
        machineLearning.loss_list,
        "loss",
        f"{save_curve_path}/loss_curve_{timestamp}.png",
    )
    saveData2Curve(
        machineLearning.acc_list, "acc", f"{save_curve_path}/acc_curve_{timestamp}.png"
    )
    # 評価
    machineLearning.evaluate()
