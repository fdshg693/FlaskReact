from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from sklearn.utils import Bunch

from util.data_handler.csv_util import read_csv_into_dataframe


class MLCompatibleDataset(BaseModel):
    """
    execute_machine_learning_pipelineなどのMLパイプラインで利用可能なデータセット形式。

    必須属性:
    - data: ２次元配列（N行のサンプル、M列の特徴量）
    - target: １次元配列（N行のターゲット（予測する値））

    例：
        data = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
        ]).
        target = np.array([0, 0, 0])  # 例えば、0はsetosaを表す

    Optional metadata (not used by the pipeline but useful for logging/UX):
    - feature_names: list of feature names
    - target_names: list of target/class names
    - descr: free-form description
    """

    data: np.ndarray  # ２次元配列（N行のサンプル、M列の特徴量）
    target: np.ndarray  # １次元配列（N行のターゲット（予測する値））
    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    descr: str | None = None

    # arbitrary_types_allowedをTrueにすることで、通常PYDANTICがサポートしていない型も許容する
    # NOTE: PYDANTICはカスタム型を検証できないため、型チェックのみを行う(例えば、np.ndarrayかどうかの確認。ただし、INT型などは通常通り検証を行う)
    # そのため、dataやtargetに対しては別途バリデーションを実装している
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ----------------------- バリデーション・型変換 -----------------------
    # インスタンス生成時のバリデーションと型変換
    @field_validator("data", mode="before")
    @classmethod
    def _coerce_data(cls, v: Any) -> np.ndarray:
        """
        dataに入る値が2D配列であることを保証する。
        その上で、float32型に変換する。
        """
        arr = np.asarray(v)
        if arr.ndim != 2:
            raise ValueError(
                f"data must be 2D array-like (n_samples, n_features); got shape {arr.shape}"
            )
        # Prefer float32 for downstream torch conversion
        return arr.astype(np.float32, copy=False)

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target(cls, v: Any) -> np.ndarray:
        """
        targetに入る値が1D配列であることを保証する。
        """
        arr = np.asarray(v).reshape(-1)
        if arr.ndim != 1:
            raise ValueError("target must be 1D array-like (n_samples,)")
        return arr

    # インスタンス生成後のバリデーション
    @field_validator("target")
    @classmethod
    def _match_lengths(cls, t: np.ndarray, info: ValidationInfo) -> np.ndarray:
        """
        インスタンス生成時のバリデーションを上で行なっているため、
        dataが２次元配列であることは保証されている。
        ここでは、dataとtargetのサンプル数（一致するか）を確認する。
        """
        data = info.data.get("data") if hasattr(info, "data") else None
        if (
            isinstance(data, np.ndarray)
            and data.ndim == 2
            and data.shape[0] != t.shape[0]
        ):
            raise ValueError(
                f"Length mismatch: len(target)={t.shape[0]} vs n_samples (data)={data.shape[0]}"
            )
        return t


def _stack_data_column(
    col: Iterable[Any], *, dtype: np.dtype | None = np.float32
) -> np.ndarray:
    """
    1次元配列要素を持つイテラブルを受け取り、2次元配列にスタックする。
    各要素は同じ長さの1次元配列でなければならない。

    例：
    col = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]
    -> np.ndarray(shape=(3, 3))
    """
    rows: list[np.ndarray] = []
    try:
        for i, item in enumerate(col):
            arr: np.ndarray = np.asarray(item)
            if arr.ndim != 1:
                raise ValueError(
                    f"Row {i}: data element must be 1D; got shape {arr.shape}"
                )
            rows.append(arr)
        stacked = np.stack(rows, axis=0)
    except Exception as exc:  # pragma: no cover - numpy provides clear error details
        raise ValueError("Failed to stack 'data' column into 2D array") from exc
    if dtype is not None:
        stacked = stacked.astype(dtype, copy=False)
    return stacked


class MLDatasetConverter:
    """
    各種データ形式をMLCompatibleDatasetに変換する統合クラス。

    サポートする入力形式:
    - pandas.DataFrame (dataとtargetカラムを持つか、featuresとtargetを指定)
    - sklearn.utils.Bunch (data, target属性を持つ)
    - CSV file path (str | Path)

    使用例:
        # DataFrameから
        ds = MLDatasetConverter.convert(df)

        # sklearn Bunchから
        ds = MLDatasetConverter.convert(sklearn_bunch)

        # CSVから
        ds = MLDatasetConverter.convert("data.csv", features=["col1", "col2"], target="label")
    """

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        *,
        data_col: str = "data",
        target_col: str = "target",
        dtype: np.dtype | None = np.float32,
    ) -> MLCompatibleDataset:
        """
        'data'と'target'カラムを持つDATAFRAMEをMLCompatibleDatasetに変換する。

        前提条件:
        - df[data_col]: 各行が1次元配列要素を持つ列
        - df[target_col]: 各行がラベルか数値からなる列

        Returns an object exposing .data and .target attributes.
        Raises ValueError on validation failure.
        """
        # datafarameに必要なカラム（data_col, target_col）が存在するか確認
        missing: list[str] = [c for c in (data_col, target_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.debug(
            "Converting DataFrame to ML dataset with columns data_col='{}', target_col='{}' (n_rows={})",
            data_col,
            target_col,
            len(df),
        )

        # dataframe.data -> 2D配列、dataframe.target -> 1D配列　に変換
        data_2d: np.ndarray = _stack_data_column(df[data_col].to_list(), dtype=dtype)
        target_1d: np.ndarray = np.asarray(df[target_col].to_numpy()).reshape(-1)

        try:
            ds = MLCompatibleDataset(data=data_2d, target=target_1d)
        except (
            ValidationError
        ) as e:  # pragma: no cover - covered by specific shape checks
            # Re-raise as ValueError to keep a simpler exception surface for callers
            raise ValueError(str(e)) from e

        logger.debug(
            "Dataset created: data shape={}, target shape={}",
            ds.data.shape,
            ds.target.shape,
        )
        return ds

    @staticmethod
    def from_sklearn_bunch(bunch: Bunch) -> MLCompatibleDataset:
        """
        sklearnスタイルのBunchオブジェクトからMLCompatibleDatasetを作成する。
        load_iris()やload_wine()などのsklearnデータセットが返すオブジェクトを、機械学習パイプラインで利用可能な形式に変換する。

        data, target属性を持つことを前提とする。
        feature_names, target_names, DESCR属性があれば、それらも利用する。

        Args:
            bunch: sklearn.utils.Bunchオブジェクト 基本はDict型と同様だが、a.bのように属性アクセスも可能
        """
        # 必須属性 data, target の存在確認＋取得
        try:
            data = bunch.data
            target = bunch.target
        except AttributeError as exc:  # pragma: no cover - trivial attr access
            raise ValueError(
                "Provided object lacks 'data' and/or 'target' attributes"
            ) from exc

        feature_names = getattr(bunch, "feature_names", None)
        target_names = getattr(bunch, "target_names", None)
        descr = getattr(bunch, "DESCR", None)

        try:
            return MLCompatibleDataset(
                data=data,
                target=target,
                feature_names=list(feature_names)
                if feature_names is not None
                else None,
                target_names=list(target_names) if target_names is not None else None,
                descr=str(descr) if descr is not None else None,
            )
        except ValidationError as e:  # pragma: no cover
            raise ValueError(str(e)) from e

    @staticmethod
    def from_csv(
        path: str | Path,
        *,
        features: list[str],
        target: str | None = None,
        encoding: str | None = None,
        dropna: bool = False,
        dtype: np.dtype | None = np.float32,
    ) -> MLCompatibleDataset:
        """
        CSVデータをMLCompatibleDatasetに変換する。

        Contract:
        - Selects given feature columns (and optional target) from the CSV.
        - Coerces features to a 2D float array (dtype arg, default float32).
        - Validates length consistency between X and y when target is provided.

        Args:
            path: CSVファイルのパス。
            features: 特徴量カラム名として扱う列名のリスト。
            target: ターゲットカラム名。Noneの場合、ターゲットなしでデータセットを作成する。
            encoding: Optional text encoding for CSV reading.
            dropna: If True, drop rows with NA across selected columns.
            dtype: Desired dtype for feature matrix.

        Returns:
            MLCompatibleDataset

        Raises:
            FileNotFoundError: If CSV path is invalid.
            ValueError: If required columns are missing or stacking fails.
            pandas errors propagated from CSV parsing.
        """
        # csvデータをDataFrameに読み込む
        df: pd.DataFrame = read_csv_into_dataframe(path, encoding=encoding)

        # 必須カラムの存在確認
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns in CSV: {missing_cols}")

        # targetカラムも確認(存在する場合)
        if target is not None and target not in df.columns:
            raise ValueError(f"Missing target column in CSV: {target}")

        sel_cols = list(features) + ([target] if target is not None else [])
        work = df.loc[:, sel_cols].copy()

        if dropna:
            before = len(work)
            work = work.dropna(axis=0, how="any")
            after = len(work)
            if before != after:
                logger.info(
                    "Dropped {} rows due to NA across selected columns", before - after
                )

        # Build X (2D) and optional y (1D)
        try:
            X = np.asarray(work[features].to_numpy())
            if X.ndim != 2:
                raise ValueError(f"Features matrix must be 2D; got {X.ndim}D")
            if dtype is not None:
                X = X.astype(dtype, copy=False)
        except Exception as exc:
            raise ValueError("Failed to build feature matrix from CSV") from exc

        if target is not None:
            y = np.asarray(work[target].to_numpy()).reshape(-1)
            ds = MLCompatibleDataset(data=X, target=y)
        else:
            # no target; create empty y with length 0 to satisfy model, or raise
            y = np.empty((0,), dtype=np.float32)
            ds = MLCompatibleDataset(data=X, target=y)

        logger.debug(
            "from_csv -> data shape={}, target shape={}",
            ds.data.shape,
            ds.target.shape,
        )
        return ds

    @staticmethod
    def convert(
        source: pd.DataFrame | Bunch | str | Path,
        *,
        # DataFrame用パラメータ
        data_col: str = "data",
        target_col: str = "target",
        # CSV用パラメータ
        features: list[str] | None = None,
        target: str | None = None,
        encoding: str | None = None,
        dropna: bool = False,
        # 共通パラメータ
        dtype: np.dtype,
    ) -> MLCompatibleDataset:
        """
        入力データの型を自動判定し、適切な変換メソッドを呼び出す。

        Args:
            source: 変換元データ (DataFrame, Bunch, CSVパス)
            data_col: DataFrameの特徴量カラム名 (DataFrame用)
            target_col: DataFrameのターゲットカラム名 (DataFrame用)
            features: CSV/DataFrameの特徴量カラム名リスト (CSV用)
            target: CSV/DataFrameのターゲットカラム名 (CSV用)
            encoding: CSV読み込み時のエンコーディング
            dropna: 欠損値を持つ行を削除するか
            dtype: 特徴量行列のデータ型

        Returns:
            MLCompatibleDataset

        Raises:
            ValueError: サポートされていない型が渡された場合、または必須パラメータが不足している場合
        """
        # pandas DataFrameの場合
        if isinstance(source, pd.DataFrame):
            logger.debug("Auto-detected DataFrame input")
            return MLDatasetConverter.from_dataframe(
                source, data_col=data_col, target_col=target_col, dtype=dtype
            )

        # sklearn Bunchの場合
        elif isinstance(source, Bunch):
            logger.debug("Auto-detected sklearn Bunch input")
            return MLDatasetConverter.from_sklearn_bunch(source)

        # CSVファイルパスの場合
        elif isinstance(source, (str, Path)):
            logger.debug("Auto-detected CSV path input")
            if features is None:
                raise ValueError(
                    "When converting from CSV, 'features' parameter is required"
                )
            return MLDatasetConverter.from_csv(
                source,
                features=features,
                target=target,
                encoding=encoding,
                dropna=dropna,
                dtype=dtype,
            )

        else:
            raise ValueError(
                f"Unsupported source type: {type(source).__name__}. "
                f"Supported types: pandas.DataFrame, sklearn.utils.Bunch, str, pathlib.Path"
            )


__all__ = [
    "MLCompatibleDataset",
    "MLDatasetConverter",
]
