# mypy の使い方

# mypyはPythonの静的型チェッカーで、コードの型チェックを行います。
# 以下に基本的な使い方を示します。

```bash
mypy your_script.py
```
# mypyの設定ファイルを作成
pyproject.toml
```toml
[mypy]
files = "src"
ignore_missing_imports = true
strict = true
```