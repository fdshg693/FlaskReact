"""CSVを入力としてプロットを作成するユーティリティ。

公開API:
    - plot_histograms_from_dataframe
    - plot_boxplots_from_dataframe
    - plot_scatter_from_dataframe

本モジュールは公開APIの表面を小さく・安定させるため、内部実装を
クラス（`CsvPlotter`）に寄せて、ヘルパー間の引数受け渡しを減らしています。

依存関係（概要）:
    公開関数 -> CsvPlotter -> _plot_* ヘルパー -> _finalize_figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from matplotlib.figure import Figure

from util.csv_plot.options import FinalizeOptions, PlotOptions
from util.csv_plot.plotter import CsvPlotter

# -----------------------------------------------------------------------------
# 公開API（安定ラッパー）
# -----------------------------------------------------------------------------


def plot_histograms_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
    close: bool | None = None,
    keep_open: bool | None = None,
) -> Dict[str, Figure]:
    """DataFrame の数値列に対してヒストグラムを作成します。

    Args:
        df: プロット対象の pandas DataFrame。
        columns: プロット対象の列名を絞り込むための任意の列名リスト。
            None の場合は数値列を自動選択します。
        bins: ヒストグラムのビン数（matplotlib に渡されます）。
        figsize: 各ヒストグラムの図のサイズ（幅, 高さ）[inch]。
        save_dir: 各ヒストグラムを PNG として保存するディレクトリ。
            指定された場合は必要に応じて作成し、`{column}_hist.png` として保存します。
        show: True の場合、各図の作成後に `plt.show()` を呼びます。
        close: 図を閉じるかどうか（`show` と独立）。None の場合は従来挙動で、
            `show=False` なら閉じ、`show=True` なら開いたままにします。
        keep_open: `close=False` の別名（`close` と同時指定不可）。

    Returns:
        作成したヒストグラムの `column name -> matplotlib.figure.Figure` の辞書。

    Notes:
        - 非数値列はスキップします。
        - NaN は除外して描画します。
    """
    options = PlotOptions(
        figsize=figsize,
        save_dir=save_dir,
        finalize=FinalizeOptions(show=show, close=close, keep_open=keep_open),
    )
    return CsvPlotter(df).plot_histograms(columns, bins=bins, options=options)


def plot_boxplots_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
    close: bool | None = None,
    keep_open: bool | None = None,
) -> Dict[str, Figure]:
    """DataFrame の数値列に対して箱ひげ図を作成します。

    `plot_histograms_from_dataframe` と同様の引数・挙動です。

    - 非数値列はスキップします。
    - NaN は除外して描画します。
    - `save_dir` 指定時は `{column}_boxplot.png` として保存します。
    """
    options = PlotOptions(
        figsize=figsize,
        save_dir=save_dir,
        finalize=FinalizeOptions(show=show, close=close, keep_open=keep_open),
    )
    return CsvPlotter(df).plot_boxplots(columns, options=options)


def plot_scatter_from_dataframe(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
    close: bool | None = None,
    keep_open: bool | None = None,
    title: Optional[str] = None,
    color: str = "C0",
    marker: str = "o",
    alpha: float = 0.8,
) -> Figure:
    """DataFrame の2列を使って散布図を作成します。

    Args:
        df: 元になる DataFrame。
        x: x軸に使う列名。
        y: y軸に使う列名。
        figsize: 図のサイズ [inch]。
        save_dir: 図の保存先ディレクトリ（任意）。
        show: True の場合 `plt.show()` を呼びます。
        close: 図を閉じるかどうか（`show` と独立）。None の場合は従来挙動で、
            `show=False` なら閉じ、`show=True` なら開いたままにします。
        keep_open: `close=False` の別名（`close` と同時指定不可）。
        title: タイトル（任意）。None の場合は既定のタイトルを使います。
        color: 点の色（`ax.scatter` に渡されます）。
        marker: マーカースタイル。
        alpha: 点の透明度。

    Returns:
        散布図を含む matplotlib の Figure。
    """
    options = PlotOptions(
        figsize=figsize,
        save_dir=save_dir,
        finalize=FinalizeOptions(show=show, close=close, keep_open=keep_open),
    )
    return CsvPlotter(df).plot_scatter(
        x,
        y,
        options=options,
        title=title,
        color=color,
        marker=marker,
        alpha=alpha,
    )


__all__ = [
    "CsvPlotter",
    "FinalizeOptions",
    "PlotOptions",
    "plot_boxplots_from_dataframe",
    "plot_histograms_from_dataframe",
    "plot_scatter_from_dataframe",
]
