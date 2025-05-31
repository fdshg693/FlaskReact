import matplotlib.pyplot as plt
import torch


def saveData2Curve(data, label_name, file_path):
    """
    データをプロットして保存する関数
    :param data: データのリスト
    :param label_name: グラフのラベル名
    :param file_path: ファイル保存先のパス
    """
    data_len = len(data)
    plt.figure()
    plt.plot(range(1, data_len + 1), data)
    plt.xlabel("Epoch")
    plt.ylabel(f"Average {label_name}")
    plt.title(f"Training {label_name} Curve")
    plt.savefig(file_path)
    plt.close()
    print(f"{file_path} に保存しました。")


def saveStudyParameter(dictionary, file_path):
    """
    学習に関するパラメータを保存する関数
    :param dictionary: 保存するパラメータの辞書
    :param file_path: 保存先のファイルパス
    """
    torch.save(dictionary, file_path)
    print(f"{file_path} に保存しました。")
