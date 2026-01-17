from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class FinalizeOptions:
    """描画後に図を表示/クローズするかを制御します。"""

    show: bool = False
    close: bool | None = None
    keep_open: bool | None = None


@dataclass(frozen=True, slots=True)
class PlotOptions:
    """各プロット種別で共通のオプションです。"""

    figsize: Tuple[float, float] = (6.0, 4.0)
    save_dir: Optional[Path | str] = None
    finalize: FinalizeOptions = field(default_factory=FinalizeOptions)
