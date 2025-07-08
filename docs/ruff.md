# RUFFの使い方
# ==================
# RUFFはPythonの静的解析ツールで、コードの品質を向上させるために使用されます。  

### .vscode
vscodeのruff拡張機能を使用することで、コードの静的解析やフォーマットを行うことができます。
設定は`.vscode/settings.json`に記述。
保存した時に、フォーマットが自動で適用されます。

ruff check <path>
ruff check --fix <path>
<path>には、相対パスを指定します。

### ruff linter
https://docs.astral.sh/ruff/linter/

### ruff formatter
https://docs.astral.sh/ruff/formatter/
