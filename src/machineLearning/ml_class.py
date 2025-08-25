"""Unified ML module with a Base class + Classification / Regression subclasses.

要望: DATASET が 連続値(回帰: Diabetes) か クラス分類(Iris) かでクラスを分離し、
共通処理を継承させるリファクタリング。

外部インターフェース互換性:
 - execute_machine_learning_pipeline(...) のシグネチャと戻り値は維持
 - accuracy_history は分類では Accuracy、回帰では R2 スコア履歴を格納
 - loss_history は両方とも学習中の損失 (分類: CrossEntropy, 回帰: MSE)
"""

from pathlib import Path
from typing import Tuple, List, Protocol

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

from .save_util import save_model_and_learning_curves_with_custom_name
from .simple_nn import SimpleNeuralNetwork


class Trainable(Protocol):  # 形式的インターフェース (将来拡張用)
    def train_model(self, epochs: int = 20) -> Tuple[List[float], List[float]]: ...
    def evaluate_model(self) -> float: ...


class BaseMLModel:
    """共通前処理 / データ分割 / スケーリング処理をまとめた基底クラス"""

    def __init__(self, dataset: object) -> None:
        self._validate_dataset(dataset)
        self.feature_data = np.asarray(dataset.data, dtype=np.float32)
        self.target = np.asarray(dataset.target)
        self.n_samples, self.n_features = self.feature_data.shape

        self.feature_scaler = StandardScaler()
        self.scaler_dir = Path(__file__).resolve().parent.parent / "scaler"
        self.scaler_dir.mkdir(exist_ok=True)

        # サブクラスで設定される
        self.neural_network_model: nn.Module | None = None
        self.loss_criterion: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        logger.info(
            f"Initialized BaseMLModel | samples={self.n_samples} features={self.n_features}"
        )

    # --------------------------- 共通ユーティリティ ---------------------------
    def _validate_dataset(self, dataset: object) -> None:
        if dataset is None:
            raise ValueError("Dataset cannot be None")
        if not hasattr(dataset, "data") or not hasattr(dataset, "target"):
            raise ValueError("Dataset must have 'data' and 'target' attributes")
        if len(dataset.data) == 0:
            raise ValueError("Dataset cannot be empty")
        if len(dataset.data) != len(dataset.target):
            raise ValueError("Data and target must have the same length")
        data_array = np.asarray(dataset.data)
        if np.any(np.isnan(data_array)):
            raise ValueError("Dataset contains NaN values")
        if np.any(np.isinf(data_array)):
            raise ValueError("Dataset contains infinite values")

    def split_train_test_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        logger.info("Splitting dataset")
        # 回帰では stratify できないので条件分岐
        stratify = None
        if self.is_classification():  # type: ignore[attr-defined]
            stratify = self.target
        self.features_train, self.features_test, self.labels_train, self.labels_test = (
            train_test_split(  # type: ignore[attr-defined]
                self.feature_data,
                self.target,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        )
        logger.info(
            f"Split done | train={len(self.features_train)} test={len(self.features_test)}"
        )

    def apply_feature_scaling(self) -> None:
        self.features_train = self.feature_scaler.fit_transform(self.features_train)
        self.features_test = self.feature_scaler.transform(self.features_test)
        logger.info("Feature scaling complete")

    def save_scaler(self, file_suffix: str) -> None:
        scaler_file_path = self.scaler_dir / f"scaler{file_suffix}.joblib"
        joblib.dump(self.feature_scaler, scaler_file_path)
        logger.info(f"Saved scaler -> {scaler_file_path}")

    # --------------------------- サブクラス実装ポイント ---------------------------
    def convert_to_tensor_datasets(self) -> None:  # pragma: no cover - abstract-ish
        raise NotImplementedError

    def create_data_loaders(self, batch_size: int = 16) -> None:
        self.training_data_loader = DataLoader(
            self.training_dataset, batch_size=batch_size, shuffle=True
        )
        self.testing_data_loader = DataLoader(
            self.testing_dataset, batch_size=batch_size, shuffle=False
        )

    # --------------------------- メトリクス (サブクラスで実装) ---------------------------
    def train_model(
        self, epochs: int = 20
    ) -> Tuple[List[float], List[float]]:  # pragma: no cover
        raise NotImplementedError

    def evaluate_model(self) -> float:  # pragma: no cover
        raise NotImplementedError

    # hint method for stratify decision
    def is_classification(self) -> bool:
        return False


class ClassificationMLModel(BaseMLModel):
    """分類タスク用モデル"""

    def __init__(self, dataset: object) -> None:  # noqa: D401
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


class RegressionMLModel(BaseMLModel):
    """回帰タスク用モデル (出力1, 損失:MSE)。

    accuracy_history には R2 スコアを格納し、evaluate_model も R2 を返す。
    """

    def __init__(self, dataset: object) -> None:
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


# --------------------------- Factory & Pipeline ---------------------------
def _decide_task_type(target: np.ndarray) -> str:
    """Very small heuristic: クラス数が 20 以下 かつ (target が整数 or 0,1,..) -> classification"""
    unique_vals = np.unique(target)
    if unique_vals.dtype.kind in {"i", "u"} and len(unique_vals) <= 20:
        return "classification"
    # 整数でも異なる値が多い場合は連続値とみなす
    if len(unique_vals) <= 10 and np.allclose(unique_vals, unique_vals.astype(int)):
        return "classification"
    return "regression"


def _build_model(dataset: object) -> BaseMLModel:
    task = _decide_task_type(np.asarray(dataset.target))
    if task == "classification":
        return ClassificationMLModel(dataset)
    return RegressionMLModel(dataset)


def execute_machine_learning_pipeline(
    dataset: object,
    epochs: int = 5,
    file_suffix: str = "",
    learning_rate: float | None = None,
) -> Tuple[BaseMLModel, nn.Module, List[float], List[float]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    logger.info(
        f"Starting machine learning pipeline | epochs={epochs} dataset_type={dataset.__class__.__name__}"
    )
    model_wrapper = _build_model(dataset)
    # 学習率オーバーライド
    if learning_rate is not None and learning_rate > 0:
        if model_wrapper.optimizer is not None:
            for pg in model_wrapper.optimizer.param_groups:
                pg["lr"] = float(learning_rate)
            logger.info(f"Overridden learning rate -> {learning_rate}")
    model_wrapper.split_train_test_data()
    model_wrapper.apply_feature_scaling()
    model_wrapper.save_scaler(file_suffix)
    model_wrapper.convert_to_tensor_datasets()
    model_wrapper.create_data_loaders()
    accuracy_history, loss_history = model_wrapper.train_model(epochs)
    logger.info("Pipeline completed")
    assert model_wrapper.neural_network_model is not None
    return (
        model_wrapper,
        model_wrapper.neural_network_model,
        accuracy_history,
        loss_history,
    )


if __name__ == "__main__":  # 簡易動作確認
    from sklearn.datasets import load_iris, load_diabetes

    iris = load_iris()
    iris_model, iris_net, iris_acc_hist, iris_loss_hist = (
        execute_machine_learning_pipeline(iris, epochs=3)
    )
    save_model_and_learning_curves_with_custom_name(
        iris_net, iris_acc_hist, iris_loss_hist, "iris", 3
    )
    logger.info(f"Iris test metric (acc) = {iris_model.evaluate_model():.4f}")

    diabetes = load_diabetes()
    diab_model, diab_net, diab_r2_hist, diab_loss_hist = (
        execute_machine_learning_pipeline(diabetes, epochs=3)
    )
    save_model_and_learning_curves_with_custom_name(
        diab_net, diab_r2_hist, diab_loss_hist, "diabetes", 3
    )
    logger.info(f"Diabetes test metric (R2) = {diab_model.evaluate_model():.4f}")
