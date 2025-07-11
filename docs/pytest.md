# pytest の使い方

pytestはPythonのテストフレームワークで、簡単にテストを作成し実行することができます。以下に基本的な使い方を示します。

## テストの作成

テストは通常、`test_`で始まる関数名を持つPythonファイル内に作成されます。例えば、`test_sample.py`というファイルを作成し、以下のように記述します。

```python
def test_addition():
    assert 1 + 1 == 2
```

## テストの実行
- ### カレントディレクトリ内のすべてのテストを実行するには、以下のコマンドを使用します。
```bash
pytest
```
- ### 特定のテストファイルを指定して実行するには、以下のようにします。
```bash
pytest test_sample.py
```
- ### 特定のテスト関数を指定して実行するには、以下のようにします。
```bash
pytest test_sample.py::test_addition
```

これにより、カレントディレクトリ内のすべてのテストが実行されます。

## より高度な機能

pytestは、フィクスチャ、マーク、パラメータ化テストなど、さまざまな高度な機能をサポートしています。詳細については、[公式ドキュメント](https://docs.pytest.org/en/stable/)を参照してください。
