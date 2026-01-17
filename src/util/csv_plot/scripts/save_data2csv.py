"""
irisデータやdiabetesデータを規定のdataフォルダ配下の場所にCSVとして保存するスクリプト
"""

from pathlib import Path
from typing import Iterable, cast

import pandas as pd
from loguru import logger
from regex import B
from sklearn.datasets import load_diabetes, load_iris
from sklearn.utils._bunch import Bunch

from config import PROJECTPATHS
from util.csv_plot.util import save_csv_to_path


def _feature_names(names: Iterable[str] | None, n_cols: int) -> list[str]:
    """
    Bunchから取得したfeature_namesを検査し、適切な列名リストを返す。
    namesがNoneの場合や要素数が不正な場合は、デフォルトの列名リストfeature_0, feature_1,...を返す。
    """
    if names:
        # 素の list[str] に正規化
        return [str(n) for n in names]
    return [f"feature_{i}" for i in range(n_cols)]


def _save_bunch_as_csv(
    data: Bunch,
    save_path: Path | str,
    *,
    include_target: bool,
    index: bool,
    header: bool | list[str] | None,
) -> Path:
    """
    data.dataとdata.feature_namesからDataFrameを構築し、必要に応じてtargetベクトルを追加し、save_csv_to_path経由で保存します。

    Args:
        data: sklearnのBunchオブジェクト（load_irisやload_diabetesの戻り値）
        save_path: 保存先のファイルパス
        include_target: target列を含めるかどうか
        index: 行名（インデックス）を書き込むかどうか
        header: 列名を書き込むかどうか、またはカスタム列名のリスト
    """

    # Bunch の内容から DataFrame を構築
    X = getattr(data, "data", None)
    if X is None:
        raise ValueError("Provided Bunch has no 'data' field")
    n_cols = X.shape[1] if hasattr(X, "shape") else len(X[0])  # type: ignore[index]
    names = _feature_names(getattr(data, "feature_names", None), int(n_cols))

    df = pd.DataFrame(data=X, columns=names)
    if include_target and hasattr(data, "target"):
        df["target"] = getattr(data, "target")

    saved = save_csv_to_path(df, save_path, index=index, header=header)
    logger.success("Saved CSV to {}", saved)
    return saved


def save_iris_to_csv(
    save_path: Path | str,
    include_target: bool = True,
    index: bool = False,
    header: bool | list[str] | None = True,
) -> Path:
    """
    irisデータセットを読み込み、CSVファイルとして保存します。
    """
    logger.info("Loading iris dataset")
    data: Bunch = cast(Bunch, load_iris())
    return _save_bunch_as_csv(
        data,
        save_path,
        include_target=include_target,
        index=index,
        header=header,
    )


def save_diabetes_to_csv(
    save_path: Path | str,
    include_target: bool = True,
    index: bool = False,
    header: bool | list[str] | None = True,
) -> Path:
    """
    diabetesデータセットを読み込み、CSVファイルとして保存します。
    """
    logger.info("Loading diabetes dataset")
    data: Bunch = cast(Bunch, load_diabetes())
    return _save_bunch_as_csv(
        data,
        save_path,
        include_target=include_target,
        index=index,
        header=header,
    )


if __name__ == "__main__":
    save_iris_to_csv(
        save_path=PROJECTPATHS.iris_data_path,
        include_target=True,
    )
    save_diabetes_to_csv(
        save_path=PROJECTPATHS.diabetes_data_path,
        include_target=True,
    )
