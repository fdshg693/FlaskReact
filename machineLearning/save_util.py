from pathlib import Path
from typing import Any, Dict, List, Literal, Union, Optional
import time

import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
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

        # Use csv module to write data directly
        with open(
            output_file_path, mode=write_mode, newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.writer(csvfile)
            for row in tabular_data:
                writer.writerow(row)

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


def save_model_and_learning_curves_with_custom_name(
    trained_model: nn.Module,
    accuracy_history: List[float],
    loss_history: List[float],
    dataset_name: str,
    epochs: int,
    file_suffix: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """
    モデルのパラメータと学習曲線をカスタム名で保存する統合関数

    Args:
        trained_model: 学習済みモデル
        accuracy_history: 精度のリスト
        loss_history: 損失のリスト
        dataset_name: データセット名
        epochs: エポック数
        custom_name: カスタムファイル名（省略時はタイムスタンプ）
        project_root: プロジェクトルートパス（省略時は自動推定）

    Returns:
        str: 使用されたタイムスタンプまたはカスタム名

    Raises:
        RuntimeError: 全ての保存操作が失敗した場合
    """
    logger.info("Saving model and learning curves")

    # タイムスタンプまたはカスタム名の決定
    file_suffix = file_suffix if file_suffix else time.strftime("%Y%m%d_%H%M%S")

    # プロジェクトルートの決定
    if project_root is None:
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent

    # 保存先ディレクトリの設定
    param_dir = project_root / "param"
    curve_log_dir = project_root / "curveLog"
    csv_log_dir = project_root / "csvLog"

    saved_files: List[str] = []
    failed_operations: List[str] = []

    # ディレクトリの作成
    for directory in [param_dir, curve_log_dir, csv_log_dir]:
        directory.mkdir(exist_ok=True)

    # モデルパラメータの保存
    try:
        model_file = param_dir / f"models_{file_suffix}.pth"
        save_training_parameters(trained_model.state_dict(), str(model_file))
        saved_files.append(str(model_file))
        logger.info(f"Model parameters saved to {model_file}")
    except Exception as e:
        error_msg = f"Failed to save model parameters: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # 学習曲線の保存
    try:
        loss_curve_file = curve_log_dir / f"loss_curve_{file_suffix}.png"
        accuracy_curve_file = curve_log_dir / f"acc_curve_{file_suffix}.png"
        save_training_data_to_curve_plot(loss_history, "loss", str(loss_curve_file))
        save_training_data_to_curve_plot(
            accuracy_history, "acc", str(accuracy_curve_file)
        )
        saved_files.extend([str(loss_curve_file), str(accuracy_curve_file)])
        logger.info(f"Learning curves saved to {curve_log_dir}")
    except Exception as e:
        error_msg = f"Failed to save learning curves: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # CSVファイルの保存
    try:
        # データの変換（エポック番号と値のペア）
        loss_data_for_csv = [
            [epoch + 1, loss_value] for epoch, loss_value in enumerate(loss_history)
        ]
        accuracy_data_for_csv = [
            [epoch + 1, accuracy_value]
            for epoch, accuracy_value in enumerate(accuracy_history)
        ]

        loss_csv_file = csv_log_dir / f"loss_{file_suffix}.csv"
        accuracy_csv_file = csv_log_dir / f"acc_{file_suffix}.csv"
        save_data_to_csv_file(loss_data_for_csv, str(loss_csv_file))
        save_data_to_csv_file(accuracy_data_for_csv, str(accuracy_csv_file))
        saved_files.extend([str(loss_csv_file), str(accuracy_csv_file)])
        logger.info(f"CSV files saved to {csv_log_dir}")
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

    # trained_model.csvに情報を追記
    try:
        trained_model_csv = project_root / "train_log" / "trained_model.csv"

        # CSVヘッダーとデータ行を準備
        csv_header = ["dataset_name", "epochs", "file_suffix", "timestamp"]
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_data_row = [dataset_name, epochs, file_suffix, current_timestamp]

        # ファイルが存在しない場合はヘッダーを追加
        if not trained_model_csv.exists():
            save_data_to_csv_file(
                [csv_header, csv_data_row], str(trained_model_csv), "w"
            )
            logger.info("Created new trained_model.csv with header and first entry")
        else:
            save_data_to_csv_file([csv_data_row], str(trained_model_csv), "a")
            logger.info("Appended entry to trained_model.csv")

    except Exception as e:
        error_msg = f"Failed to update trained_model.csv: {e}"
        logger.error(error_msg)
        # CSVファイルの更新失敗は全体の処理を止めない

    return file_suffix
