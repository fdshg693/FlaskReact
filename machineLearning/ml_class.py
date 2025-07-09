from pathlib import Path
import time
from typing import Tuple, List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
import numpy as np

from .save_util import (
    save_training_data_to_curve_plot,
    save_training_parameters,
    save_data_to_csv_file,
)


class SimpleNeuralNetwork(nn.Module):
    """
    シンプルなニューラルネットワークモデル

    2層の全結合層で構成された分類用ニューラルネットワーク
    """

    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 16,
        output_dimension: int = 3,
    ) -> None:
        """
        ニューラルネットワークの初期化

        Args:
            input_dimension: 入力層の次元数
            hidden_dimension: 隠れ層の次元数
            output_dimension: 出力層の次元数
        """
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(input_dimension, hidden_dimension)
        self.fully_connected_layer_2 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        順伝播の実行

        Args:
            input_tensor: 入力テンソル

        Returns:
            torch.Tensor: 出力テンソル
        """
        hidden_output = F.relu(self.fully_connected_layer_1(input_tensor))
        return self.fully_connected_layer_2(hidden_output)


class MachineLearningClassifier:
    def __init__(self, dataset: object) -> None:
        """
        機械学習分類器のコンストラクタ

        Args:
            dataset: sklearnのデータセットオブジェクト
        """
        # Critical: Validate dataset to prevent runtime errors
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not hasattr(dataset, "data") or not hasattr(dataset, "target"):
            raise ValueError("Dataset must have 'data' and 'target' attributes")
        if len(dataset.data) == 0:
            raise ValueError("Dataset cannot be empty")
        if len(dataset.data) != len(dataset.target):
            raise ValueError("Data and target must have the same length")

        # Critical: Check for NaN or infinite values that would cause training to fail
        data_array = np.array(dataset.data)
        if np.any(np.isnan(data_array)):
            raise ValueError(
                "Dataset contains NaN values which will cause training to fail"
            )
        if np.any(np.isinf(data_array)):
            raise ValueError(
                "Dataset contains infinite values which will cause training to fail"
            )

        logger.info("Initializing MachineLearningClassifier")
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
        logger.info(f"Dataset loaded with {len(self.feature_data)} samples")

    def split_train_test_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        """
        訓練データとテストデータに分割する

        Args:
            test_size: テストデータの割合
            random_state: 乱数シード値
        """
        logger.info(
            f"Splitting data with test_size={test_size}, random_state={random_state}"
        )
        self.features_train, self.features_test, self.labels_train, self.labels_test = (
            train_test_split(
                self.feature_data,
                self.target_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=self.target_labels,
            )
        )
        logger.info(
            f"Training set size: {len(self.features_train)}, Test set size: {len(self.features_test)}"
        )

    def apply_feature_scaling(self) -> None:
        """特徴量の標準化を適用する"""
        logger.info("Applying feature scaling")
        self.features_train = self.feature_scaler.fit_transform(self.features_train)
        self.features_test = self.feature_scaler.transform(self.features_test)
        logger.info("Feature scaling completed")

    def convert_to_tensor_datasets(self) -> None:
        """データをTensorDatasetに変換する"""
        logger.info("Converting data to tensor datasets")
        self.training_dataset = TensorDataset(
            torch.tensor(self.features_train, dtype=torch.float32),
            torch.tensor(self.labels_train, dtype=torch.long),
        )
        self.testing_dataset = TensorDataset(
            torch.tensor(self.features_test, dtype=torch.float32),
            torch.tensor(self.labels_test, dtype=torch.long),
        )
        logger.info("Tensor datasets created")

    def create_data_loaders(self) -> None:
        """データローダーを作成する"""
        logger.info("Creating data loaders")
        self.training_data_loader = DataLoader(
            self.training_dataset, batch_size=16, shuffle=True
        )
        self.testing_data_loader = DataLoader(
            self.testing_dataset, batch_size=16, shuffle=False
        )
        logger.info("Data loaders created")

    def train_model(self, epochs: int = 20) -> Tuple[List[float], List[float]]:
        """
        モデルを訓練する

        Args:
            epochs: 学習エポック数

        Returns:
            Tuple[List[float], List[float]]: 精度履歴と損失履歴のタプル
        """
        # Critical: Validate epochs to prevent runtime errors
        if epochs < 1:
            raise ValueError("epochs must be at least 1")

        logger.info(f"Starting model training for {epochs} epochs")
        training_loss_history: List[float] = []
        training_accuracy_history: List[float] = []

        for epoch in range(1, epochs + 1):
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

            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}, Accuracy: {average_accuracy:.4f}"
                )

        logger.info("Model training completed")
        return training_accuracy_history, training_loss_history

    def evaluate_model(self) -> float:
        """
        モデルを評価する

        Returns:
            float: テスト精度
        """
        logger.info("Evaluating model on test data")
        self.neural_network_model.eval()
        correct_predictions = 0

        with torch.no_grad():
            for feature_batch, label_batch in self.testing_data_loader:
                predicted_labels = self.neural_network_model(feature_batch).argmax(
                    dim=1
                )
                correct_predictions += (predicted_labels == label_batch).sum().item()

        test_accuracy = correct_predictions / len(self.testing_dataset)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        return test_accuracy


def execute_machine_learning_pipeline(
    dataset: object, epochs: int = 20
) -> Tuple[MachineLearningClassifier, nn.Module, List[float], List[float]]:
    """
    機械学習の実行をまとめた関数

    Args:
        dataset: sklearnのデータセットオブジェクト
        epochs: 学習エポック数

    Returns:
        Tuple containing: classifier, neural network model, accuracy history, loss history
    """
    # Critical: Validate epochs parameter to prevent runtime errors
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    logger.info(f"Starting machine learning pipeline with {epochs} epochs")
    machine_learning_classifier = MachineLearningClassifier(dataset)
    machine_learning_classifier.split_train_test_data()
    machine_learning_classifier.apply_feature_scaling()
    machine_learning_classifier.convert_to_tensor_datasets()
    machine_learning_classifier.create_data_loaders()
    accuracy_history, loss_history = machine_learning_classifier.train_model(epochs)
    logger.info("Machine learning pipeline completed")
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

    Args:
        trained_model: 学習済みモデル
        accuracy_history: 精度のリスト
        loss_history: 損失のリスト
    """
    logger.info("Saving model and learning curves")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_file_path = Path(__file__).resolve()

    saved_files: List[str] = []
    failed_operations: List[str] = []

    # パラメータの保存
    try:
        parameter_save_path = current_file_path.parent.parent / "param"
        parameter_save_path.mkdir(exist_ok=True)
        parameter_file = parameter_save_path / f"models_{timestamp}.pth"
        save_training_parameters(trained_model.state_dict(), str(parameter_file))
        saved_files.append(str(parameter_file))
        logger.info(f"Model parameters saved to {parameter_file}")
    except Exception as e:
        error_msg = f"Failed to save model parameters: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # 学習曲線の保存
    try:
        curve_save_path = current_file_path.parent.parent / "curveLog"
        curve_save_path.mkdir(exist_ok=True)
        loss_curve_file = curve_save_path / f"loss_curve_{timestamp}.png"
        accuracy_curve_file = curve_save_path / f"acc_curve_{timestamp}.png"
        save_training_data_to_curve_plot(loss_history, "loss", str(loss_curve_file))
        save_training_data_to_curve_plot(
            accuracy_history, "acc", str(accuracy_curve_file)
        )
        saved_files.extend([str(loss_curve_file), str(accuracy_curve_file)])
        logger.info(f"Learning curves saved to {curve_save_path}")
    except Exception as e:
        error_msg = f"Failed to save learning curves: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # CSVファイルの保存
    try:
        csv_save_path = current_file_path.parent.parent / "csvLog"
        csv_save_path.mkdir(exist_ok=True)
        # Convert single lists to list of lists format expected by saveData2CSV
        loss_data_for_csv = [
            [epoch + 1, loss_value] for epoch, loss_value in enumerate(loss_history)
        ]
        accuracy_data_for_csv = [
            [epoch + 1, accuracy_value]
            for epoch, accuracy_value in enumerate(accuracy_history)
        ]
        loss_csv_file = csv_save_path / f"loss_{timestamp}.csv"
        accuracy_csv_file = csv_save_path / f"acc_{timestamp}.csv"
        save_data_to_csv_file(loss_data_for_csv, str(loss_csv_file))
        save_data_to_csv_file(accuracy_data_for_csv, str(accuracy_csv_file))
        saved_files.extend([str(loss_csv_file), str(accuracy_csv_file)])
        logger.info(f"CSV files saved to {csv_save_path}")
    except Exception as e:
        error_msg = f"Failed to save CSV files: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # 結果の報告
    if saved_files:
        logger.info(f"Successfully saved {len(saved_files)} files: {saved_files}")

    if failed_operations:
        error_summary = f"Some save operations failed: {failed_operations}"
        logger.error(error_summary)
        if not saved_files:  # 全て失敗した場合のみ例外を発生
            raise RuntimeError(error_summary)
        else:
            logger.warning(
                f"Partial save completed. {len(saved_files)} files saved, {len(failed_operations)} operations failed."
            )


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    logger.info("Starting machine learning experiment")

    # Irisデータセットでの実行
    logger.info("Loading Iris dataset")
    iris_dataset = load_iris()
    iris_classifier, iris_neural_network, iris_accuracy_history, iris_loss_history = (
        execute_machine_learning_pipeline(iris_dataset)
    )
    save_model_and_learning_curves(
        iris_neural_network, iris_accuracy_history, iris_loss_history
    )
    iris_test_accuracy = iris_classifier.evaluate_model()
    logger.info(f"Iris Test Accuracy: {iris_test_accuracy:.3f}")
    print(f"Iris Test Accuracy: {iris_test_accuracy:.3f}")

    # # Diabetesデータセットでの実行
    # diabetes_dataset = load_diabetes()
    # diabetes_classifier, diabetes_neural_network, diabetes_accuracy_history, diabetes_loss_history = execute_machine_learning_pipeline(diabetes_dataset)
    # save_model_and_learning_curves(diabetes_neural_network, diabetes_accuracy_history, diabetes_loss_history)
    # print(f"Diabetes Test Accuracy: {diabetes_classifier.evaluate_model():.3f}")
