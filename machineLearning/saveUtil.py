from pathlib import Path
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch


def saveData2Curve(
    data: List[float], label_name: str, file_path: Union[str, Path]
) -> None:
    """
    データをプロットして保存する関数
    :param data: データのリスト
    :param label_name: グラフのラベル名
    :param file_path: ファイル保存先のパス
    """
    file_path = Path(file_path)
    data_len = len(data)
    plt.figure()
    plt.plot(range(1, data_len + 1), data)
    plt.xlabel("Epoch")
    plt.ylabel(f"Average {label_name}")
    plt.title(f"Training {label_name} Curve")
    plt.savefig(file_path)
    plt.close()
    print(f"{file_path} に保存しました。")


def saveData2CSV(
    data: List[List[Union[str, float, int]]],
    file_path: Union[str, Path],
    mode: Literal["w", "a"] = "w",
) -> None:
    """
    データをCSV形式で保存する関数
    :param data: 保存するデータのリスト
    :param file_path: 保存先のファイルパス
    :param mode: ファイルの書き込みモード ('w' for write, 'a' for append)
    """
    file_path = Path(file_path)
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, mode=mode)
    print(f"{file_path} に保存しました。")


def saveStudyParameter(dictionary: dict, file_path: Union[str, Path]) -> None:
    """
    学習に関するパラメータを保存する関数
    :param dictionary: 保存するパラメータの辞書
    :param file_path: 保存先のファイルパス
    """
    file_path = Path(file_path)
    torch.save(dictionary, file_path)
    print(f"{file_path} に保存しました。")
