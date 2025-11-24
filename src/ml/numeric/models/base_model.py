from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

from numeric.dataset import MLCompatibleDataset
from config import ensure_path_exists


class BaseMLModel:
    """共通前処理 / データ分割 / スケーリング処理をまとめた基底クラス"""

    def __init__(self, dataset: MLCompatibleDataset) -> None:
        self._validate_dataset(dataset)
        self.feature_data = np.asarray(dataset.data, dtype=np.float32)
        self.target = np.asarray(dataset.target)
        self.n_samples, self.n_features = self.feature_data.shape

        self.feature_scaler = StandardScaler()

        # サブクラスで設定される
        self.neural_network_model: Optional[nn.Module] = None
        self.loss_criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Subclasses should set these
        self.training_dataset = None
        self.testing_dataset = None
        self.training_data_loader = None
        self.testing_data_loader = None

        # Split data placeholders
        self.features_train = None
        self.features_test = None
        self.labels_train = None
        self.labels_test = None

        logger.info(
            f"Initialized BaseMLModel | samples={self.n_samples} features={self.n_features}"
        )

    # --------------------------- 共通ユーティリティ ---------------------------
    def _validate_dataset(self, dataset: MLCompatibleDataset) -> None:
        """
        以下のことを確認:
        - dataset が None でない
        - dataset に data, target 属性がある
        - data, target の長さが同じ
        - data が空でない
        - data に NaN や無限大が含まれていない
        """
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

    def save_scaler(self, save_dir: Path) -> None:
        """
        Arguments:
            save_dir: 保存ディレクトリパス
        """
        # スケーラー保存ディレクトリがあることを確認
        ensure_path_exists(save_dir)
        scaler_file_path = save_dir / "scaler.joblib"
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
