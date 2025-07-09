from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger


def save_training_data_to_curve_plot(
    training_data: List[float], metric_label: str, output_file_path: Union[str, Path]
) -> None:
    """
    データをプロットして保存する関数
    :param training_data: 学習データのリスト
    :param metric_label: グラフのメトリクスラベル名
    :param output_file_path: ファイル保存先のパス
    """
    # Input validation
    if not training_data:
        raise ValueError("Training data cannot be empty")

    if not isinstance(metric_label, str) or not metric_label.strip():
        raise ValueError("Metric label must be a non-empty string")

    output_file_path = Path(output_file_path)

    try:
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create plot with proper resource management
        fig, ax = plt.subplots(figsize=(10, 6))
        total_epochs = len(training_data)
        ax.plot(range(1, total_epochs + 1), training_data)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Average {metric_label}")
        ax.set_title(f"Training {metric_label} Curve")
        fig.savefig(output_file_path)

        logger.info(f"{output_file_path} に保存しました。")

    except Exception as e:
        logger.error(f"Failed to save plot to {output_file_path}: {e}")
        raise
    finally:
        plt.close(fig)  # Ensure cleanup even if error occurs


def save_data_to_csv_file(
    tabular_data: List[List[Union[str, float, int]]],
    output_file_path: Union[str, Path],
    write_mode: Literal["w", "a"] = "w",
) -> None:
    """
    データをCSV形式で保存する関数
    :param tabular_data: 保存するテーブル形式データのリスト
    :param output_file_path: 保存先のファイルパス
    :param write_mode: ファイルの書き込みモード ('w' for write, 'a' for append)
    """
    # Input validation
    if not tabular_data:
        raise ValueError("Tabular data cannot be empty")

    # Validate data structure consistency
    if len(set(len(row) for row in tabular_data)) > 1:
        raise ValueError("All rows must have the same number of columns")

    output_file_path = Path(output_file_path)

    try:
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        data_frame = pd.DataFrame(tabular_data)
        data_frame.to_csv(output_file_path, index=False, mode=write_mode)
        logger.info(f"{output_file_path} に保存しました。")

    except Exception as e:
        logger.error(f"Failed to save CSV to {output_file_path}: {e}")
        raise


def save_training_parameters(
    parameter_dictionary: Dict[str, Any], output_file_path: Union[str, Path]
) -> None:
    """
    学習に関するパラメータを保存する関数
    :param parameter_dictionary: 保存する学習パラメータの辞書
    :param output_file_path: 保存先のファイルパス
    """
    # Input validation
    if not parameter_dictionary:
        raise ValueError("Parameter dictionary cannot be empty")

    output_file_path = Path(output_file_path)

    try:
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(parameter_dictionary, output_file_path)
        logger.info(f"{output_file_path} に保存しました。")

    except Exception as e:
        logger.error(f"Failed to save parameters to {output_file_path}: {e}")
        raise
