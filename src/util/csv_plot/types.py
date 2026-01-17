from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Protocol

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# matplotlib の描画関数は有用な戻り値を返すことがあります
# （例: `ax.hist` -> tuple、`ax.boxplot` -> dict）。
# 本プロジェクトではそれらの戻り値を使わないため、コールバックの戻り値型は
# 特定せず `object` としています。
PlotFn = Callable[[Axes, pd.Series, Dict[str, Any]], object]


# 注:
# Matplotlib（および一部の pandas）の型スタブは、いくつかのメソッドシグネチャに
# `Unknown` を含みます（`**kwargs: Unknown` など）。その結果 `ax.set_title` や
# `fig.savefig` といったメンバー参照で Pyright の reportUnknownMemberType が
# 発生しやすくなります。
# ここで定義する小さな Protocol に cast することで、グローバルに抑制せずに
# 既知の API 面だけを扱えるようにしています。


class _AxesLike(Protocol):
    def hist(self, x: Any, bins: int = ..., **kwargs: Any) -> Any: ...

    def boxplot(self, x: Any, **kwargs: Any) -> Any: ...

    def scatter(self, x: Any, y: Any, **kwargs: Any) -> Any: ...

    def set_title(self, label: str, **kwargs: Any) -> Any: ...

    def set_xlabel(self, xlabel: str, **kwargs: Any) -> Any: ...

    def set_ylabel(self, ylabel: str, **kwargs: Any) -> Any: ...


class _FigureLike(Protocol):
    def savefig(self, fname: str | Path, **kwargs: Any) -> None: ...


class _DataFrameDropna(Protocol):
    def dropna(self, *args: Any, **kwargs: Any) -> pd.DataFrame: ...
