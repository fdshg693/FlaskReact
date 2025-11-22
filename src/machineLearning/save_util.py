from pathlib import Path
from typing import Any, Dict, List, Literal, Union, Optional
import time

import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
from loguru import logger
from config import PATHS


def store_model_and_learning_logs(
    trained_model: nn.Module,
    accuracy_history: List[float],
    loss_history: List[float],
    dataset_name: str,
    epochs: int,
    experiment_name: Optional[str] = None,
    log_dirs_root: Optional[Path] = None,
) -> str:
    """
    モデルのパラメータと学習曲線・学習CSVログ・学習済みモデル情報を保存する統合関数

    作成されるフォルダ構成:
    {log_dirs_root}/{experiment_name}/
        ├── model_param.pth
        ├── loss_curve.png
        ├── acc_curve.png
        ├── loss.csv
        └── acc.csv

    また、{log_dirs_root}/train_log/trained_model.csv に学習情報を追記します。

    Args:
        trained_model: 学習済みモデル
        accuracy_history: 精度のリスト
        loss_history: 損失のリスト
        dataset_name: データセット名
        epochs: エポック数
        experiment_name: 実験名（フォルダ名）。省略時はタイムスタンプを使用。
        log_dirs_root: 保存先ディレクトリのルートパス（省略時はデフォルトパス）。

    Returns:
        str: 使用された実験名（フォルダ名）

    Raises:
        RuntimeError: 全ての保存操作が失敗した場合
    """
    logger.info("Saving model and learning curves")

    # タイムスタンプまたはカスタム名の決定
    experiment_name = (
        experiment_name if experiment_name else time.strftime("%Y%m%d_%H%M%S")
    )

    # プロジェクトルートの決定
    if log_dirs_root is None:
        log_dirs_root = PATHS.ml_outputs

    # 保存先ディレクトリの設定 (実験ごとのフォルダを作成)
    experiment_dir = log_dirs_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []
    failed_operations: List[str] = []

    # モデルパラメータの保存
    try:
        model_file = experiment_dir / "model_param.pth"
        _save_training_parameters(trained_model.state_dict(), str(model_file))
        saved_files.append(str(model_file))
        logger.info(f"Model parameters saved to {model_file}")
    except Exception as e:
        error_msg = f"Failed to save model parameters: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # 学習曲線の保存
    try:
        loss_curve_file = experiment_dir / "loss_curve.png"
        accuracy_curve_file = experiment_dir / "acc_curve.png"
        _save_training_curve_2_png(loss_history, "loss", str(loss_curve_file))
        _save_training_curve_2_png(accuracy_history, "acc", str(accuracy_curve_file))
        saved_files.extend([str(loss_curve_file), str(accuracy_curve_file)])
        logger.info(f"Learning curves saved to {experiment_dir}")
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

        loss_csv_file = experiment_dir / "loss.csv"
        accuracy_csv_file = experiment_dir / "acc.csv"
        _save_csv_data(loss_data_for_csv, str(loss_csv_file))
        _save_csv_data(accuracy_data_for_csv, str(accuracy_csv_file))
        saved_files.extend([str(loss_csv_file), str(accuracy_csv_file)])
        logger.info(f"CSV files saved to {experiment_dir}")
    except Exception as e:
        error_msg = f"Failed to save CSV files: {e}"
        failed_operations.append(error_msg)
        logger.error(error_msg)

    # 結果の報告
    if saved_files:
        logger.info(f"Successfully saved {len(saved_files)} files in {experiment_dir}")

    if failed_operations:
        error_summary = f"Some save operations failed: {failed_operations}"
        logger.error(error_summary)
        if not saved_files:  # 全て失敗した場合のみ例外を発生
            raise RuntimeError(error_summary)

    # trained_model.csvに情報を追記
    try:
        # train_log フォルダはルート直下に作成（集計用）
        train_log_dir = log_dirs_root / "train_log"
        train_log_dir.mkdir(exist_ok=True)
        trained_model_csv = train_log_dir / "trained_model.csv"

        # CSVヘッダーとデータ行を準備
        csv_header = ["dataset_name", "epochs", "experiment_name", "timestamp"]
        current_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_data_row = [dataset_name, epochs, experiment_name, current_timestamp]

        # ファイルが存在しない場合はヘッダーを追加
        if not trained_model_csv.exists():
            _save_csv_data([csv_header, csv_data_row], str(trained_model_csv), "w")
            logger.info("Created new trained_model.csv with header and first entry")
        else:
            _save_csv_data([csv_data_row], str(trained_model_csv), "a")
            logger.info("Appended entry to trained_model.csv")

    except Exception as e:
        error_msg = f"Failed to update trained_model.csv: {e}"
        logger.error(error_msg)
        # CSVファイルの更新失敗は全体の処理を止めない

    return experiment_name


# ===========================
# 内部ユーティリティ関数
# ===========================


# loss.png や acc.png を保存する関数
def _save_training_curve_2_png(
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


# loss.csv や acc.csv そして、 trained_model.csv を保存する関数
def _save_csv_data(
    tabular_data: List[List[Union[str, float, int]]],
    output_file_path: Union[str, Path],
    write_mode: Literal["w", "a"] = "w",
) -> None:
    """
    データをCSV形式で保存する関数

    Args:
        tabular_data: 2次元リスト形式の学習データ
        output_file_path: 保存先のファイルパス
        write_mode: 書き込みモード ("w"=上書き, "a"=追記)
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


# model_param.pth を保存する関数
def _save_training_parameters(
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
