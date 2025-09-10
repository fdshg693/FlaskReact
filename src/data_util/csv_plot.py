from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from config import TMP_DIR, DIABETES_DATA_PATH, get_path
from data_util.csv_util import read_csv_from_path


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
    if columns is not None:
        cols = [c for c in columns if c in df.columns]
    else:
        cols = df.select_dtypes(include="number").columns.tolist()

    logger.debug("plot_histograms_from_dataframe() -> plotting columns: {}", cols)

    figs: Dict[str, plt.Figure] = {}

    out_dir: Optional[Path] = Path(save_dir) if save_dir is not None else None
    if out_dir is not None and not out_dir.exists():
        logger.debug("Creating save directory: {}", out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        ser = df[col].dropna()
        if ser.empty:
            logger.warning("Column %s is empty after dropping NaNs; skipping.", col)
            continue

        fig, ax = plt.subplots(figsize=figsize)
        try:
            ax.hist(ser, bins=bins)
            ax.set_title(f"Histogram: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")

            figs[col] = fig

            if out_dir is not None:
                safe_name = str(col).replace(" ", "_")
                out_path = out_dir / f"{safe_name}_hist.png"
                fig.savefig(out_path, bbox_inches="tight")
                logger.info("Saved histogram for %s to %s", col, out_path)

            if show:
                plt.show()
            else:
                plt.close(fig)
        except Exception:
            # close figure on error to avoid memory leak
            plt.close(fig)
            logger.exception("Failed to create histogram for column %s", col)
            raise

    return figs


if __name__ == "__main__":
    # df_example = read_csv_from_path(
    #     IRIS_DATA_PATH,
    #     encoding="utf-8",
    # )

    df_example = read_csv_from_path(
        DIABETES_DATA_PATH,
        encoding="utf-8",
    )

    dir_path = get_path("outputs", root=TMP_DIR, create=True)

    plot_histograms_from_dataframe(
        df_example,
        bins=10,
        figsize=(8, 6),
        save_dir=dir_path,
        show=False,
    )
