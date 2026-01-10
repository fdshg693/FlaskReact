from __future__ import annotations

"""
    IRIS CSVデータを読み込み、散布図をプロットするデモ。
"""

from config import PROJECTPATHS, get_path
from util.csv_plot.main import plot_scatter_from_dataframe
from util.csv_plot.util import read_csv_into_dataframe


def main() -> None:
    df_example = read_csv_into_dataframe(
        PROJECTPATHS.iris_data_path,
        encoding="utf-8",
    )

    out_dir = get_path("csv_plot_demo", root=PROJECTPATHS.tmp, create=True)

    plot_scatter_from_dataframe(
        df_example,
        x="sepal length (cm)",
        y="sepal width (cm)",
        figsize=(8, 6),
        save_dir=out_dir,
        show=False,
        title="Iris Dataset: Sepal Length vs Sepal Width",
        color="C1",
        marker="o",
        alpha=0.7,
    )


if __name__ == "__main__":
    main()
