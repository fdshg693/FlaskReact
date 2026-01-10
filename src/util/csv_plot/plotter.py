from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, cast

import pandas as pd
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.pyplot import close as mpl_close
from matplotlib.pyplot import show as mpl_show
from matplotlib.pyplot import subplots as mpl_subplots

from config import ensure_path_exists
from util.csv_plot.options import PlotOptions
from util.csv_plot.types import (
    PlotFn,
    _AxesLike,
    _DataFrameDropna,
    _FigureLike,
)


def _resolve_should_close(
    *, show: bool, close: bool | None, keep_open: bool | None
) -> bool:
    if close is not None and keep_open is not None:
        raise ValueError("Specify only one of close= or keep_open=")

    if close is not None:
        return close
    if keep_open is not None:
        return not keep_open

    # 後方互換の既定挙動: これまでは `show=False` なら図を閉じ、`show=True` なら
    # 図を開いたままにしていました。
    return not show


def _finalize_figure(
    fig: Figure,
    *,
    show: bool,
    close: bool | None,
    keep_open: bool | None,
) -> None:
    """matplotlib の Figure を後処理します。

    仕様:
        - `show=True` の場合 `plt.show()` を呼びます。
        - クローズの有無は `close` / `keep_open` で `show` と独立に制御します。
        - `close` と `keep_open` のどちらも指定されない場合は従来挙動に合わせ、
          `show=False` なら閉じ、`show=True` なら開いたままにします。
    """

    should_close = _resolve_should_close(show=show, close=close, keep_open=keep_open)
    if show:
        mpl_show()
    if should_close:
        mpl_close(fig)


def _select_plot_columns(
    df: pd.DataFrame, columns: Optional[Iterable[str]]
) -> list[str]:
    if columns is not None:
        return [c for c in columns if c in df.columns]
    return df.select_dtypes(include="number").columns.tolist()


def _sanitize_filename_component(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")


def _resolve_out_dir(save_dir: Optional[Path | str]) -> Path | None:
    if save_dir is None:
        return None
    return ensure_path_exists(Path(save_dir))


class CsvPlotter:
    """DataFrame を入力としてプロットを作成する実装クラスです。"""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def plot_histograms(
        self,
        columns: Optional[Iterable[str]] = None,
        *,
        bins: int = 30,
        options: PlotOptions,
    ) -> Dict[str, Figure]:
        return self._plot_series_columns(
            columns,
            options=options,
            title_prefix="Histogram",
            save_suffix="hist",
            plot_fn=lambda ax, s, kw: cast(_AxesLike, ax).hist(s, bins=bins, **kw),
            plot_kwargs={},
            ylabel="Count",
        )

    def plot_boxplots(
        self,
        columns: Optional[Iterable[str]] = None,
        *,
        options: PlotOptions,
    ) -> Dict[str, Figure]:
        return self._plot_series_columns(
            columns,
            options=options,
            title_prefix="Boxplot",
            save_suffix="boxplot",
            plot_fn=lambda ax, s, kw: cast(_AxesLike, ax).boxplot(s, **kw),
            plot_kwargs={},
            ylabel=None,
        )

    def plot_scatter(
        self,
        x: str,
        y: str,
        *,
        options: PlotOptions,
        title: Optional[str] = None,
        color: str = "C0",
        marker: str = "o",
        alpha: float = 0.8,
    ) -> Figure:
        df = self._df

        if x not in df.columns:
            raise KeyError(f"x column '{x}' not found in DataFrame")
        if y not in df.columns:
            raise KeyError(f"y column '{y}' not found in DataFrame")

        data = cast(_DataFrameDropna, df[[x, y]]).dropna()
        if data.empty:
            logger.warning(
                "No rows remain after dropping NaNs for columns {} and {}",
                x,
                y,
            )
            raise ValueError("No data to plot after dropping NaNs")

        fig, ax = mpl_subplots(figsize=options.figsize)
        ax_like = cast(_AxesLike, ax)
        fig_like = cast(_FigureLike, fig)
        try:
            ax_like.scatter(data[x], data[y], c=color, marker=marker, alpha=alpha)
            ax_like.set_xlabel(x)
            ax_like.set_ylabel(y)
            ax_like.set_title(title or f"Scatter: {x} vs {y}")

            out_dir = _resolve_out_dir(options.save_dir)
            if out_dir is not None:
                safe_x = _sanitize_filename_component(str(x))
                safe_y = _sanitize_filename_component(str(y))
                out_path = out_dir / f"{safe_x}_{safe_y}_scatter.png"
                fig_like.savefig(out_path, bbox_inches="tight")
                logger.info("Saved scatter for {} vs {} to {}", x, y, out_path)

            _finalize_figure(
                fig,
                show=options.finalize.show,
                close=options.finalize.close,
                keep_open=options.finalize.keep_open,
            )
            return fig
        except Exception:
            mpl_close(fig)
            logger.exception("Failed to create scatter plot for {} vs {}", x, y)
            raise

    def _plot_series_columns(
        self,
        columns: Optional[Iterable[str]],
        *,
        options: PlotOptions,
        title_prefix: str,
        save_suffix: str,
        plot_fn: PlotFn,
        plot_kwargs: Dict[str, Any],
        ylabel: Optional[str] = None,
    ) -> Dict[str, Figure]:
        df = self._df
        cols = _select_plot_columns(df, columns)
        logger.debug("CsvPlotter._plot_series_columns() -> plotting columns: {}", cols)

        figs: Dict[str, Figure] = {}
        for col in cols:
            fig = self._plot_one_series_column(
                col,
                options=options,
                title_prefix=title_prefix,
                save_suffix=save_suffix,
                plot_fn=plot_fn,
                plot_kwargs=plot_kwargs,
                ylabel=ylabel,
            )
            if fig is not None:
                figs[col] = fig
        return figs

    def _plot_one_series_column(
        self,
        col: str,
        *,
        options: PlotOptions,
        title_prefix: str,
        save_suffix: str,
        plot_fn: PlotFn,
        plot_kwargs: Dict[str, Any],
        ylabel: Optional[str],
    ) -> Optional[Figure]:
        df = self._df
        ser = df[col].dropna()
        if ser.empty:
            logger.warning("Column {} is empty after dropping NaNs; skipping.", col)
            return None

        fig, ax = mpl_subplots(figsize=options.figsize)
        ax_like = cast(_AxesLike, ax)
        fig_like = cast(_FigureLike, fig)
        try:
            plot_fn(ax, ser, plot_kwargs)
            ax_like.set_title(f"{title_prefix}: {col}" if title_prefix else str(col))
            ax_like.set_xlabel(str(col))
            if ylabel:
                ax_like.set_ylabel(ylabel)

            out_dir = _resolve_out_dir(options.save_dir)
            if out_dir is not None:
                safe_name = _sanitize_filename_component(str(col))
                out_path = out_dir / f"{safe_name}_{save_suffix}.png"
                fig_like.savefig(out_path, bbox_inches="tight")
                logger.info("Saved {} for {} to {}", save_suffix, col, out_path)

            _finalize_figure(
                fig,
                show=options.finalize.show,
                close=options.finalize.close,
                keep_open=options.finalize.keep_open,
            )
            return fig
        except Exception:
            mpl_close(fig)
            logger.exception("Failed to create {} for column {}", save_suffix, col)
            raise
