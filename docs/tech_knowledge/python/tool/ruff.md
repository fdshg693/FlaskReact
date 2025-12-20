# RUFFの使い方
# ==================
# RUFFはPythonの静的解析ツールで、コードの品質を向上させるために使用されます。  

### .vscode
vscodeのruff拡張機能を使用することで、コードの静的解析やフォーマットを行うことができます。
設定は`.vscode/settings.json`に記述。
保存した時に、フォーマットが自動で適用されます。

- ruffのチェックを実行
ruff check `path`
- ruffの自動修正を実行
ruff check --fix `path`
- ruffでimport整理のみを実行
ruff check --select I --fix `path`

`path`には、相対パスを指定します。`.`はプロジェクトルートを指します。

### pyproject.toml
```toml
[tool.ruff]
line-length = 88
exclude = ["tests/*"]

[tool.ruff.lint]
select = ["I"]  # インポートのソートを有効化
```


### ruff linter
https://docs.astral.sh/ruff/linter/

### ruff formatter
https://docs.astral.sh/ruff/formatter/
