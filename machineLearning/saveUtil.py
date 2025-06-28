from pathlib import Path
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch


def save_training_data_to_curve_plot(
    training_data: List[float], metric_label: str, output_file_path: Union[str, Path]
) -> None:
    """
    データをプロットして保存する関数
    :param training_data: 学習データのリスト
    :param metric_label: グラフのメトリクスラベル名
    :param output_file_path: ファイル保存先のパス
    """
    output_file_path = Path(output_file_path)
    total_epochs = len(training_data)
    plt.figure()
    plt.plot(range(1, total_epochs + 1), training_data)
    plt.xlabel("Epoch")
    plt.ylabel(f"Average {metric_label}")
    plt.title(f"Training {metric_label} Curve")
    plt.savefig(output_file_path)
    plt.close()
    print(f"{output_file_path} に保存しました。")


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
    output_file_path = Path(output_file_path)
    data_frame = pd.DataFrame(tabular_data)
    data_frame.to_csv(output_file_path, index=False, mode=write_mode)
    print(f"{output_file_path} に保存しました。")


def save_training_parameters(
    parameter_dictionary: dict, output_file_path: Union[str, Path]
) -> None:
    """
    学習に関するパラメータを保存する関数
    :param parameter_dictionary: 保存する学習パラメータの辞書
    :param output_file_path: 保存先のファイルパス
    """
    output_file_path = Path(output_file_path)
    torch.save(parameter_dictionary, output_file_path)
    print(f"{output_file_path} に保存しました。")
