import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch.nn as nn
from loguru import logger

from config import PROJECTPATHS
from ml.numeric.dataset import MLCompatibleDataset
from ml.numeric.models.base_model import BaseMLModel
from ml.numeric.models.classification_model import ClassificationMLModel
from ml.numeric.models.regression_model import RegressionMLModel
from ml.numeric.save_util import store_model_and_learning_logs


# --------------------------- Factory & Pipeline ---------------------------
def _decide_task_type(target: np.ndarray) -> str:
    """Very small heuristic: クラス数が 20 以下 かつ (target が整数 or 0,1,..) -> classification"""
    unique_vals = np.unique(target)
    if unique_vals.dtype.kind in {"i", "u"} and len(unique_vals) <= 20:  # pyright: ignore[reportUnknownMemberType]
        return "classification"
    # 整数でも異なる値が多い場合は連続値とみなす
    if len(unique_vals) <= 10 and np.allclose(unique_vals, unique_vals.astype(int)):
        return "classification"
    return "regression"


def _build_model(dataset: MLCompatibleDataset) -> BaseMLModel:
    task = _decide_task_type(np.asarray(dataset.target))
    if task == "classification":
        return ClassificationMLModel(dataset)
    return RegressionMLModel(dataset)


def execute_machine_learning_pipeline(
    dataset: MLCompatibleDataset,
    epochs: int = 5,
    learning_rate: Optional[float] = None,
    experiment_name: Optional[str] = None,
    log_dirs_root: Optional[Path] = None,
) -> Tuple[BaseMLModel, nn.Module, List[float], List[float], str]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    logger.info(
        f"Starting machine learning pipeline | epochs={epochs} dataset_type={dataset.__class__.__name__}"
    )

    # 実験名と保存先の設定
    if experiment_name is None:
        experiment_name = time.strftime("%Y%m%d_%H%M%S")

    if log_dirs_root is None:
        log_dirs_root = PROJECTPATHS.ml_logs

    save_dir = log_dirs_root / experiment_name

    model_wrapper = _build_model(dataset)
    # 学習率オーバーライド
    if learning_rate is not None and learning_rate > 0:
        if model_wrapper.optimizer is not None:
            for pg in model_wrapper.optimizer.param_groups:
                pg["lr"] = float(learning_rate)
            logger.info(f"Overridden learning rate -> {learning_rate}")
    model_wrapper.split_train_test_data()
    model_wrapper.apply_feature_scaling()
    model_wrapper.save_scaler(save_dir)
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
        experiment_name,
    )


def train_and_save_pipeline(
    dataset: MLCompatibleDataset,
    dataset_name: str,
    epochs: int = 5,
    learning_rate: Optional[float] = None,
    experiment_name: Optional[str] = None,
    log_dirs_root: Optional[Path] = None,
) -> Tuple[BaseMLModel, nn.Module, List[float], List[float], str]:
    """
    Executes the training pipeline and saves the model and logs.
    This combines execute_machine_learning_pipeline and store_model_and_learning_logs.
    """
    model_wrapper, net, acc_hist, loss_hist, exp_name = (
        execute_machine_learning_pipeline(
            dataset=dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
            log_dirs_root=log_dirs_root,
        )
    )

    store_model_and_learning_logs(
        trained_model=net,
        accuracy_history=acc_hist,
        loss_history=loss_hist,
        dataset_name=dataset_name,
        epochs=epochs,
        experiment_name=exp_name,
    )

    return model_wrapper, net, acc_hist, loss_hist, exp_name
