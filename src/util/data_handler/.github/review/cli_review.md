# src/util.data_handler モジュール コードレビュー

## 総合評価: 8/10

`src/util.data_handler`モジュールは、CSV操作とデータ可視化のための実用的なユーティリティを提供しています。モダンなPythonのベストプラクティス（`pathlib`、型ヒント、`loguru`）に概ね準拠しており、コードの品質は高いレベルにあります。

---

## 優先度の高い改善点

### 1. 【優先度: 高】型ヒントの一貫性とPydantic活用の欠如

**該当ファイル:** `csv_plot.py`（67-69行目）

**問題点:**
```python
plot_fn: Callable[[plt.Axes, pd.Series, Dict[str, Any]], None] = lambda ax,
s,
kw: None,
```
- 関数定義が複数行に分割され、可読性が低下
- `Pydantic`を用いた設定の検証が行われていない（プロジェクト標準では推奨）

**改善提案:**
```python
from pydantic import BaseModel, Field

class PlotConfig(BaseModel):
    """Plot configuration settings."""
    figsize: Tuple[float, float] = Field(default=(6.0, 4.0), description="Figure size in inches")
    bins: int = Field(default=30, ge=1, description="Number of histogram bins")
    show: bool = Field(default=False, description="Display plot immediately")

# 関数シグネチャを整理
PlotFunction = Callable[[plt.Axes, pd.Series, Dict[str, Any]], None]

def _plot_columns_from_dataframe(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    config: PlotConfig = PlotConfig(),
    plot_fn: PlotFunction = lambda ax, s, kw: None,
    # ...
) -> Dict[str, plt.Figure]:
```

---

### 2. 【優先度: 高】エラーハンドリングの粒度不足

**該当ファイル:** `csv_util.py`（41-43行目）、`csv_plot.py`（119-122行目）

**問題点:**
```python
except Exception:  # keep broad to let callers handle pandas exceptions
    logger.exception("Failed to read CSV from %s", p)
    raise
```
- 広範な`Exception`キャッチは、予期しないエラーを隠蔽する可能性がある
- 特定のエラータイプに対する適切な処理が欠けている

**改善提案:**
```python
from pandas.errors import EmptyDataError, ParserError

try:
    df = pd.read_csv(p, encoding=encoding) if encoding else pd.read_csv(p)
    logger.info("Loaded CSV with shape {} from {}", df.shape, p)
    return df
except FileNotFoundError as e:
    logger.error("CSV file not found at path: {}", p)
    raise
except EmptyDataError as e:
    logger.error("CSV file is empty: {}", p)
    raise ValueError(f"Empty CSV file: {p}") from e
except ParserError as e:
    logger.error("Failed to parse CSV: {}", p)
    raise ValueError(f"Invalid CSV format: {p}") from e
except Exception as e:
    logger.exception("Unexpected error reading CSV from %s", p)
    raise RuntimeError(f"Failed to read CSV: {p}") from e
```

---

### 3. 【優先度: 中】未使用コードとコメントアウトの残存

**該当ファイル:** `csv_plot.py`（228-245行目）

**問題点:**
```python
if __name__ == "__main__":
    df_example = read_csv_into_dataframe(
        PATHS.iris_data_path,
        encoding="utf-8",
    )

    # df_example = read_csv_from_path(
    #     DIABETES_DATA_PATH,
    #     encoding="utf-8",
    # )

    dir_path = get_path("outputs", root=PATHS.tmp, create=True)

    # plot_boxplots_from_dataframe(
    #     df_example,
    #     figsize=(8, 6),
    #     save_dir=dir_path,
    #     show=False,
    # )
```
- コメントアウトされたコードが複数残存
- 実行例としては不明瞭で、保守性を低下させる

**改善提案:**
```python
# examples/ ディレクトリに適切な例示ファイルを作成するか、削除する
# 例: src/util.data_handler/examples/plot_examples.py
```

---

### 4. 【優先度: 中】docstringの一貫性

**該当ファイル:** `csv_plot.py`（73-76行目）

**問題点:**
```python
def _plot_columns_from_dataframe(
    # ...
) -> Dict[str, plt.Figure]:
    """Common plotting helper used by concrete plot functions.

    plot_fn is a callable that draws on the given Axes. It receives (ax, series, plot_kwargs).
    """
```
- 内部ヘルパー関数にも関わらず、引数の説明が不足している
- 他の関数のdocstring形式（Args/Returns）との一貫性がない

**改善提案:**
```python
def _plot_columns_from_dataframe(
    # ...
) -> Dict[str, plt.Figure]:
    """Common plotting helper used by concrete plot functions.
    
    Args:
        df: Source DataFrame.
        columns: Optional column names to plot.
        figsize: Figure size in inches.
        save_dir: Optional directory to save figures.
        show: Whether to display plots.
        title_prefix: Prefix for plot titles.
        save_suffix: Suffix for saved filenames.
        plot_fn: Callable that draws on the Axes. Receives (ax, series, plot_kwargs).
        plot_kwargs: Additional keyword arguments passed to plot_fn.
        ylabel: Optional y-axis label.
    
    Returns:
        Dictionary mapping column names to matplotlib Figure objects.
    """
```

---

### 5. 【優先度: 低】命名規則の微調整

**該当ファイル:** `examples/save_data2csv.py`

**問題点:**
- ファイル名が`save_data2csv.py`（2を使用）
- Python命名規則では、数字を英単語にするか、アンダースコアで区切るのが一般的

**改善提案:**
```
save_data_to_csv.py
```

---

## 良好な点

1. **モダンPythonの採用**: `pathlib.Path`、型ヒント、`loguru`を適切に使用
2. **関数の単一責任**: 各関数が明確な責務を持ち、再利用可能
3. **ドキュメンテーション**: 主要な関数に詳細なdocstringを提供
4. **柔軟性**: オプショナル引数により、様々なユースケースに対応
5. **ロギング**: 適切なレベルでのログ出力が実装されている

---

## 推奨アクション

1. `Pydantic`を用いた設定クラスの導入（優先度: 高）
2. 例外処理の粒度を細かく分割（優先度: 高）
3. コメントアウトコードの整理・削除（優先度: 中）
4. 内部関数のdocstring充実化（優先度: 中）
5. ファイル名の命名規則統一（優先度: 低）

---

**レビュー実施日:** 2025-11-23  
**レビュー対象:** `src/util.data_handler/` モジュール全体
