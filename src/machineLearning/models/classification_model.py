from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from loguru import logger

from machineLearning.models.base_model import BaseMLModel
from machineLearning.simple_nn import SimpleNeuralNetwork
from machineLearning.dataset import MLCompatibleDataset


class ClassificationMLModel(BaseMLModel):
    """分類タスク用モデル"""

    def __init__(self, dataset: MLCompatibleDataset) -> None:  # noqa: D401
        super().__init__(dataset)
        # クラス数を推定
        unique_classes = np.unique(self.target)
        self.n_classes = int(len(unique_classes))
        self.neural_network_model = SimpleNeuralNetwork(
            input_dim=self.n_features, hidden_dim=16, output_dim=self.n_classes
        )
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.neural_network_model.parameters(), lr=0.1, momentum=0.9
        )
        logger.info(f"Classification model initialized | classes={self.n_classes}")

    def is_classification(self) -> bool:  # override
        return True

    def convert_to_tensor_datasets(self) -> None:
        logger.info("Creating tensor datasets (classification)")
        self.training_dataset = TensorDataset(
            torch.tensor(self.features_train, dtype=torch.float32),
            torch.tensor(self.labels_train, dtype=torch.long),
        )
        self.testing_dataset = TensorDataset(
            torch.tensor(self.features_test, dtype=torch.float32),
            torch.tensor(self.labels_test, dtype=torch.long),
        )

    def train_model(self, epochs: int = 20) -> Tuple[List[float], List[float]]:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        assert self.neural_network_model and self.loss_criterion and self.optimizer
        acc_history: List[float] = []
        loss_history: List[float] = []
        for epoch in range(1, epochs + 1):
            self.neural_network_model.train()
            total_loss = 0.0
            correct = 0
            for xb, yb in self.training_data_loader:
                self.optimizer.zero_grad()
                logits = self.neural_network_model(xb)
                loss = self.loss_criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
            avg_loss = total_loss / len(self.training_dataset)
            acc = correct / len(self.training_dataset)
            loss_history.append(avg_loss)
            acc_history.append(acc)
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                logger.info(
                    f"[Cls] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} acc={acc:.4f}"
                )
        return acc_history, loss_history

    def evaluate_model(self) -> float:
        assert self.neural_network_model
        self.neural_network_model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in self.testing_data_loader:
                pred = self.neural_network_model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
        acc = correct / len(self.testing_dataset)
        logger.info(f"Classification test accuracy={acc:.4f}")
        return acc
