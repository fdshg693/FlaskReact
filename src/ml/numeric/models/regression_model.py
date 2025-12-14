from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import TensorDataset

from ml.numeric.dataset import MLCompatibleDataset
from ml.numeric.models.base_model import BaseMLModel
from ml.numeric.simple_nn import SimpleNeuralNetwork


class RegressionMLModel(BaseMLModel):
    """回帰タスク用モデル (出力1, 損失:MSE)。

    accuracy_history には R2 スコアを格納し、evaluate_model も R2 を返す。
    """

    def __init__(self, dataset: MLCompatibleDataset) -> None:
        super().__init__(dataset)
        self.neural_network_model = SimpleNeuralNetwork(
            input_dim=self.n_features, hidden_dim=32, output_dim=1
        )
        self.loss_criterion = nn.MSELoss()
        # 回帰は勾配が発散しやすいので学習率を小さく
        self.optimizer = torch.optim.SGD(
            self.neural_network_model.parameters(), lr=0.01, momentum=0.9
        )
        logger.info("Regression model initialized | output_dim=1")

    def convert_to_tensor_datasets(self) -> None:
        y_train = self.labels_train.astype(np.float32)
        y_test = self.labels_test.astype(np.float32)
        # shape (N,) -> (N,1)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        self.training_dataset = TensorDataset(
            torch.tensor(self.features_train, dtype=torch.float32), y_train_t
        )
        self.testing_dataset = TensorDataset(
            torch.tensor(self.features_test, dtype=torch.float32), y_test_t
        )

    def _r2_score(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        if ss_tot.item() == 0:  # 全て同じ値の場合
            return 0.0
        return 1 - (ss_res / ss_tot).item()

    def train_model(self, epochs: int = 20) -> Tuple[List[float], List[float]]:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        assert self.neural_network_model and self.loss_criterion and self.optimizer
        r2_history: List[float] = []
        loss_history: List[float] = []
        for epoch in range(1, epochs + 1):
            self.neural_network_model.train()
            total_loss = 0.0
            for xb, yb in self.training_data_loader:
                self.optimizer.zero_grad()
                preds = self.neural_network_model(xb)
                loss = self.loss_criterion(preds, yb)
                loss.backward()
                # 勾配爆発対策
                torch.nn.utils.clip_grad_norm_(
                    self.neural_network_model.parameters(), max_norm=5.0
                )
                self.optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(self.training_dataset)
            # R2 (train) 計算
            self.neural_network_model.eval()
            with torch.no_grad():
                all_preds = []
                all_targets = []
                for xb, yb in self.training_data_loader:
                    pr = self.neural_network_model(xb)
                    all_preds.append(pr)
                    all_targets.append(yb)
                y_pred_cat = torch.cat(all_preds, dim=0)
                y_true_cat = torch.cat(all_targets, dim=0)
                r2 = self._r2_score(y_true_cat, y_pred_cat)
            loss_history.append(avg_loss)
            r2_history.append(r2)
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                logger.info(
                    f"[Reg] Epoch {epoch}/{epochs} | loss={avg_loss:.4f} R2={r2:.4f}"
                )
        return r2_history, loss_history

    def evaluate_model(self) -> float:
        assert self.neural_network_model
        self.neural_network_model.eval()
        with torch.no_grad():
            preds_list = []
            targets_list = []
            for xb, yb in self.testing_data_loader:
                preds_list.append(self.neural_network_model(xb))
                targets_list.append(yb)
            y_pred = torch.cat(preds_list, dim=0)
            y_true = torch.cat(targets_list, dim=0)
            r2 = self._r2_score(y_true, y_pred)
        logger.info(f"Regression test R2={r2:.4f}")
        return r2
