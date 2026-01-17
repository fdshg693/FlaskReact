"""CSV plotting tests.

Run normally:
    pytest -q tests/test_csv_plot.py

Run visual artifact tests (writes PNGs under PROJECTPATHS.tmp -> data/tmp/csv_plot_manual):
    RUN_VISUAL_TESTS=1 pytest -q -m visual -s tests/test_csv_plot.py
"""

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from config import PROJECTPATHS, get_path
from util.csv_plot.main import (
    plot_boxplots_from_dataframe,
    plot_histograms_from_dataframe,
    plot_scatter_from_dataframe,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 2, 3, None],
            "b": [10.0, 10.5, 11.0, None, 12.0],
            "c": ["x", "y", "z", "x", "y"],
        }
    )


def _assert_numeric_plots_created(
    out: dict[str, object], save_dir: Path, *, suffix: str
) -> None:
    assert "a" in out
    assert "b" in out

    a_path = save_dir / f"a_{suffix}.png"
    b_path = save_dir / f"b_{suffix}.png"
    assert a_path.exists()
    assert b_path.exists()


def _visual_enabled(request: pytest.FixtureRequest) -> bool:
    if os.getenv("RUN_VISUAL_TESTS") == "1":
        return True
    markexpr = str(getattr(request.config.option, "markexpr", "") or "")
    return "visual" in markexpr


@pytest.mark.parametrize(
    ("plot_fn", "suffix"),
    [
        (plot_histograms_from_dataframe, "hist"),
        (plot_boxplots_from_dataframe, "boxplot"),
    ],
)
def test_plot_numeric_columns_tmpdir(
    sample_df: pd.DataFrame,
    tmp_path: Path,
    plot_fn: Callable[..., dict[str, object]],
    suffix: str,
) -> None:
    out = plot_fn(sample_df, save_dir=tmp_path)
    _assert_numeric_plots_created(out, tmp_path, suffix=suffix)


def test_plot_scatter_saves_png(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    fig = plot_scatter_from_dataframe(sample_df, x="a", y="b", save_dir=tmp_path)
    assert fig is not None
    out_path = tmp_path / "a_b_scatter.png"
    assert out_path.exists()


def test_plot_scatter_missing_columns_raises(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        plot_scatter_from_dataframe(sample_df, x="missing", y="b")
    with pytest.raises(KeyError):
        plot_scatter_from_dataframe(sample_df, x="a", y="missing")


@pytest.mark.visual
@pytest.mark.parametrize(
    ("plot_fn", "suffix"),
    [
        (plot_histograms_from_dataframe, "hist"),
        (plot_boxplots_from_dataframe, "boxplot"),
    ],
)
def test_plot_numeric_columns_projectpaths_tmp_manual(
    request: pytest.FixtureRequest,
    sample_df: pd.DataFrame,
    plot_fn: Callable[..., dict[str, object]],
    suffix: str,
) -> None:
    if not _visual_enabled(request):
        pytest.skip(
            "visual artifact test disabled (set RUN_VISUAL_TESTS=1 or run with -m visual)"
        )

    out_dir = get_path("csv_plot_manual", root=PROJECTPATHS.tmp, create=True)
    out = plot_fn(sample_df, save_dir=out_dir)
    _assert_numeric_plots_created(out, out_dir, suffix=suffix)
    print(f"csv_plot manual artifacts written to: {out_dir}")
