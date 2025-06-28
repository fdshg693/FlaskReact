from pathlib import Path
import time
from typing import Tuple, List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from saveUtil import (
    save_training_data_to_curve_plot,
    save_training_parameters,
    save_data_to_csv_file,
)


class SimpleNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 16,
        output_dimension: int = 3,
    ) -> None:
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(input_dimension, hidden_dimension)
        self.fully_connected_layer_2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_output = F.relu(self.fully_connected_layer_1(input_tensor))
        return self.fully_connected_layer_2(hidden_output)


class MachineLearningClassifier:
    def __init__(self, dataset) -> None:
        """
        コンストラクタ
        :param dataset: sklearnのデータセットオブジェクト
        """
        # モデル、損失関数、スケーラー、オプティマイザの初期化
        self.neural_network_model = SimpleNeuralNetwork()
        self.loss_criterion = nn.CrossEntropyLoss()
        self.feature_scaler = StandardScaler()
        self.optimizer = torch.optim.SGD(
            self.neural_network_model.parameters(), lr=0.1, momentum=0.9
        )
        # データセットの読み込み
        self.feature_data = dataset.data
        self.target_labels = dataset.target

    def split_train_test_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        self.features_train, self.features_test, self.labels_train, self.labels_test = (
            train_test_split(
                self.feature_data,
                self.target_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=self.target_labels,
            )
        )

    def apply_feature_scaling(self) -> None:
        self.features_train = self.feature_scaler.fit_transform(self.features_train)
        self.features_test = self.feature_scaler.transform(self.features_test)

    def convert_to_tensor_datasets(self) -> None:
        self.training_dataset = TensorDataset(
            torch.tensor(self.features_train, dtype=torch.float32),
            torch.tensor(self.labels_train, dtype=torch.long),
        )
        self.testing_dataset = TensorDataset(
            torch.tensor(self.features_test, dtype=torch.float32),
            torch.tensor(self.labels_test, dtype=torch.long),
        )

    def create_data_loaders(self) -> None:
        self.training_data_loader = DataLoader(
            self.training_dataset, batch_size=16, shuffle=True
        )
        self.testing_data_loader = DataLoader(
            self.testing_dataset, batch_size=16, shuffle=False
        )

    def train_model(self, epochs: int = 20) -> Tuple[List[float], List[float]]:
        training_loss_history: List[float] = []
        training_accuracy_history: List[float] = []
        for _ in range(1, epochs + 1):
            self.neural_network_model.train()
            total_loss = 0.0
            correct_predictions = 0
            for feature_batch, label_batch in self.training_data_loader:
                self.optimizer.zero_grad()
                model_outputs = self.neural_network_model(feature_batch)
                batch_loss = self.loss_criterion(model_outputs, label_batch)
                batch_loss.backward()
                self.optimizer.step()
                # 損失を計算
                total_loss += batch_loss.item() * feature_batch.size(0)
                # 予測ラベルと正解ラベルを比較して正解数を集計
                predicted_labels = model_outputs.argmax(dim=1)
                correct_predictions += (predicted_labels == label_batch).sum().item()

            # 平均損失を計算
            average_loss = total_loss / len(self.training_dataset)
            training_loss_history.append(average_loss)
            # 平均精度を計算
            average_accuracy = correct_predictions / len(self.training_dataset)
            training_accuracy_history.append(average_accuracy)
        return training_accuracy_history, training_loss_history

    def evaluate_model(self) -> float:
        self.neural_network_model.eval()
        correct_predictions = 0
        with torch.no_grad():
            for feature_batch, label_batch in self.testing_data_loader:
                predicted_labels = self.neural_network_model(feature_batch).argmax(
                    dim=1
                )
                correct_predictions += (predicted_labels == label_batch).sum().item()

        test_accuracy = correct_predictions / len(self.testing_dataset)
        return test_accuracy


def execute_machine_learning_pipeline(dataset, epochs: int = 20):
    """
    機械学習の実行をまとめた関数
    :param dataset: sklearnのデータセットオブジェクト
    :param epochs: 学習エポック数
    """
    machine_learning_classifier = MachineLearningClassifier(dataset)
    machine_learning_classifier.split_train_test_data()
    machine_learning_classifier.apply_feature_scaling()
    machine_learning_classifier.convert_to_tensor_datasets()
    machine_learning_classifier.create_data_loaders()
    accuracy_history, loss_history = machine_learning_classifier.train_model(epochs)
    return (
        machine_learning_classifier,
        machine_learning_classifier.neural_network_model,
        accuracy_history,
        loss_history,
    )


def save_model_and_learning_curves(
    trained_model: nn.Module, accuracy_history: List[float], loss_history: List[float]
) -> None:
    """
    モデルのパラメータと学習曲線を保存する関数
    :param trained_model: 学習済みモデル
    :param accuracy_history: 精度のリスト
    :param loss_history: 損失のリスト
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_file_path = Path(__file__).resolve()

    parameter_save_path = current_file_path.parent / "../param"
    save_training_parameters(
        trained_model.state_dict(), str(parameter_save_path / f"models_{timestamp}.pth")
    )

    curve_save_path = current_file_path.parent / "../curveLog"
    save_training_data_to_curve_plot(
        loss_history,
        "loss",
        str(curve_save_path / f"loss_curve_{timestamp}.png"),
    )
    save_training_data_to_curve_plot(
        accuracy_history, "acc", str(curve_save_path / f"acc_curve_{timestamp}.png")
    )

    csv_save_path = current_file_path.parent / "../csvLog"
    # Convert single lists to list of lists format expected by saveData2CSV
    loss_data_for_csv = [
        [epoch + 1, loss_value] for epoch, loss_value in enumerate(loss_history)
    ]
    accuracy_data_for_csv = [
        [epoch + 1, accuracy_value]
        for epoch, accuracy_value in enumerate(accuracy_history)
    ]
    save_data_to_csv_file(
        loss_data_for_csv, str(csv_save_path / f"loss_{timestamp}.csv")
    )
    save_data_to_csv_file(
        accuracy_data_for_csv, str(csv_save_path / f"acc_{timestamp}.csv")
    )


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Irisデータセットでの実行
    iris_dataset = load_iris()
    iris_classifier, iris_neural_network, iris_accuracy_history, iris_loss_history = (
        execute_machine_learning_pipeline(iris_dataset)
    )
    save_model_and_learning_curves(
        iris_neural_network, iris_accuracy_history, iris_loss_history
    )
    print(f"Iris Test Accuracy: {iris_classifier.evaluate_model():.3f}")

    # # Diabetesデータセットでの実行
    # diabetes_dataset = load_diabetes()
    # diabetes_classifier, diabetes_neural_network, diabetes_accuracy_history, diabetes_loss_history = execute_machine_learning_pipeline(diabetes_dataset)
    # save_model_and_learning_curves(diabetes_neural_network, diabetes_accuracy_history, diabetes_loss_history)
    # print(f"Diabetes Test Accuracy: {diabetes_classifier.evaluate_model():.3f}")
