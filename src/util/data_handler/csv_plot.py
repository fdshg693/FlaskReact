from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from config import PROJECTPATHS, get_path
from util.data_handler.csv_util import read_csv_into_dataframe


def plot_histograms_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    bins: int = 30,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """Create histograms for numeric columns in a DataFrame.

    Args:
            df: pandas DataFrame containing the data to plot.
            columns: Optional iterable of column names to restrict which columns to
                    plot. If None, all numeric columns will be used.
            bins: Number of histogram bins (passed to matplotlib).
            figsize: Figure size for each histogram (width, height) in inches.
            save_dir: Optional directory path to save each histogram PNG. If
                    provided the directory will be created if needed and each figure
                    saved as `{column}_hist.png`.
            show: If True, call `plt.show()` after creating each figure. For
                    programmatic use keep False.

    Returns:
            A dict mapping column name -> matplotlib.figure.Figure for each
            histogram created.

    Notes:
            - Non-numeric columns are skipped.
            - NaN values are ignored when plotting.
    """
    return _plot_columns_from_dataframe(
        df,
        columns,
        figsize=figsize,
        save_dir=save_dir,
        show=show,
        title_prefix="Histogram",
        save_suffix="hist",
        plot_fn=lambda ax, s, kw: ax.hist(s, bins=bins, **kw),
        plot_kwargs={},
        ylabel="Count",
    )


def _plot_columns_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
    title_prefix: str = "",
    save_suffix: str = "",
    plot_fn: Callable[[plt.Axes, pd.Series, Dict[str, Any]], None] = lambda ax,
    s,
    kw: None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    ylabel: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """Common plotting helper used by concrete plot functions.

    plot_fn is a callable that draws on the given Axes. It receives (ax, series, plot_kwargs).
    """
    if columns is not None:
        cols = [c for c in columns if c in df.columns]
    else:
        cols = df.select_dtypes(include="number").columns.tolist()

    logger.debug("_plot_columns_from_dataframe() -> plotting columns: {}", cols)

    figs: Dict[str, plt.Figure] = {}

    out_dir: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if out_dir is not None and not out_dir.exists():
        logger.debug("Creating save directory: {}", out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    plot_kwargs = plot_kwargs or {}

    for col in cols:
        ser = df[col].dropna()
        if ser.empty:
            logger.warning(f"Column {col} is empty after dropping NaNs; skipping.")
            continue

        fig, ax = plt.subplots(figsize=figsize)
        try:
            plot_fn(ax, ser, plot_kwargs)
            ax.set_title(f"{title_prefix}: {col}" if title_prefix else str(col))
            ax.set_xlabel(col)
            if ylabel:
                ax.set_ylabel(ylabel)

            figs[col] = fig

            if out_dir is not None:
                safe_name = str(col).replace(" ", "_")
                out_path = out_dir / f"{safe_name}_{save_suffix}.png"
                fig.savefig(out_path, bbox_inches="tight")
                logger.info(f"Saved {save_suffix} for {col} to {out_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception:
            plt.close(fig)
            logger.exception(f"Failed to create {save_suffix} for column {col}")
            raise

    return figs


def plot_boxplots_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    *,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """Create boxplots for numeric columns in a DataFrame.

    Mirrors the behavior and arguments of `plot_histograms_from_dataframe`.

    - Non-numeric columns are skipped.
    - NaN values are ignored when plotting.
    - Each figure is saved as `{column}_boxplot.png` when `save_dir` is provided.
    """
    return _plot_columns_from_dataframe(
        df,
        columns,
        figsize=figsize,
        save_dir=save_dir,
        show=show,
        title_prefix="Boxplot",
        save_suffix="boxplot",
        plot_fn=lambda ax, s, kw: ax.boxplot(s, **kw),
        plot_kwargs={},
    )


def plot_scatter_from_dataframe(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    figsize: Tuple[float, float] = (6.0, 4.0),
    save_dir: Optional[Path | str] = None,
    show: bool = False,
    title: Optional[str] = None,
    color: str = "C0",
    marker: str = "o",
    alpha: float = 0.8,
) -> plt.Figure:
    """Create a scatter plot for two columns in a DataFrame.

    Args:
        df: Source DataFrame.
        x: Column name for x-axis.
        y: Column name for y-axis.
        figsize: Figure size in inches.
        save_dir: Optional directory to save the figure.
        show: If True, call ``plt.show()``; otherwise the figure is closed.
        title: Optional title. If None a default is used.
        color: Color for points (passed to ``ax.scatter``).
        marker: Marker style for points.
        alpha: Alpha transparency for points.

    Returns:
        The matplotlib Figure object containing the scatter plot.
    """
    if x not in df.columns:
        raise KeyError(f"x column '{x}' not found in DataFrame")
    if y not in df.columns:
        raise KeyError(f"y column '{y}' not found in DataFrame")

    data = df[[x, y]].dropna()
    if data.empty:
        logger.warning("No rows remain after dropping NaNs for columns %s and %s", x, y)
        raise ValueError("No data to plot after dropping NaNs")

    fig, ax = plt.subplots(figsize=figsize)
    try:
        ax.scatter(data[x], data[y], c=color, marker=marker, alpha=alpha)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f"Scatter: {x} vs {y}")

        out_dir: Optional[Path] = Path(save_dir) if save_dir is not None else None
        if out_dir is not None and not out_dir.exists():
            logger.debug("Creating save directory: {}", out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        if out_dir is not None:
            safe_x = str(x).replace(" ", "_")
            safe_y = str(y).replace(" ", "_")
            out_path = out_dir / f"{safe_x}_{safe_y}_scatter.png"
            fig.savefig(out_path, bbox_inches="tight")
            logger.info("Saved scatter for %s vs %s to %s", x, y, out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
    except Exception:
        plt.close(fig)
        logger.exception("Failed to create scatter plot for %s vs %s", x, y)
        raise


if __name__ == "__main__":
    df_example = read_csv_into_dataframe(
        PROJECTPATHS.iris_data_path,
        encoding="utf-8",
    )

    # df_example = read_csv_from_path(
    #     DIABETES_DATA_PATH,
    #     encoding="utf-8",
    # )

    dir_path = get_path("outputs", root=PROJECTPATHS.tmp, create=True)

    # plot_boxplots_from_dataframe(
    #     df_example,
    #     figsize=(8, 6),
    #     save_dir=dir_path,
    #     show=False,
    # )

    # sepal length (cm),sepal width (cm)をプロット
    plot_scatter_from_dataframe(
        df_example,
        x="sepal length (cm)",
        y="sepal width (cm)",
        figsize=(8, 6),
        save_dir=dir_path,
        show=False,
        title="Iris Dataset: Sepal Length vs Sepal Width",
        color="C1",
        marker="o",
        alpha=0.7,
    )
