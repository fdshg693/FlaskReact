from pathlib import Path

import pandas as pd

from util.data_handler.csv_plot import plot_histograms_from_dataframe


def test_plot_histograms_tmpdir(tmp_path: Path):
    # build a small DataFrame with numeric and non-numeric columns
    df = pd.DataFrame(
        {
            "a": [1, 2, 2, 3, None],
            "b": [10.0, 10.5, 11.0, None, 12.0],
            "c": ["x", "y", "z", "x", "y"],
        }
    )

    out = plot_histograms_from_dataframe(df, save_dir=tmp_path)

    # expect histograms for numeric columns a and b
    assert "a" in out
    assert "b" in out

    # files should exist on disk
    a_path = tmp_path / "a_hist.png"
    b_path = tmp_path / "b_hist.png"
    assert a_path.exists()
    assert b_path.exists()


def test_plot_boxplots_tmpdir(tmp_path: Path):
    # build a small DataFrame with numeric and non-numeric columns
    df = pd.DataFrame(
        {
            "a": [1, 2, 2, 3, None],
            "b": [10.0, 10.5, 11.0, None, 12.0],
            "c": ["x", "y", "z", "x", "y"],
        }
    )

    from util.data_handler.csv_plot import plot_boxplots_from_dataframe

    out = plot_boxplots_from_dataframe(df, save_dir=tmp_path)

    # expect boxplots for numeric columns a and b
    assert "a" in out
    assert "b" in out

    # files should exist on disk
    a_path = tmp_path / "a_boxplot.png"
    b_path = tmp_path / "b_boxplot.png"
    assert a_path.exists()
    assert b_path.exists()
