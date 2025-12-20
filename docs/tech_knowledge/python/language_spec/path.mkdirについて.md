# `Path.mkdir(parents=True)` の詳細解説

## 基本シグネチャ

```python
Path.mkdir(mode=0o777, parents=False, exist_ok=False)
```

## 各パラメータの動作

### `parents=True` の効果

このフラグは、**中間ディレクトリを再帰的に作成するかどうか**を制御します。

```python
from pathlib import Path

path = Path("/tmp/a/b/c/d")

# parents=False (デフォルト): 親ディレクトリが存在しないとFileNotFoundError
path.mkdir()  # FileNotFoundError: [Errno 2] No such file or directory

# parents=True: 存在しない中間ディレクトリをすべて作成
path.mkdir(parents=True)  # /tmp/a, /tmp/a/b, /tmp/a/b/c, /tmp/a/b/c/d を順に作成
```

### 内部実装の仕組み

CPythonの実装を見ると、`parents=True`の場合は以下のような処理が行われます：

```python
# 簡略化した内部ロジック（実際のCPython実装に基づく）
def mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        os.mkdir(self, mode)
    except FileNotFoundError:
        if not parents or self.parent == self:
            raise
        # 再帰的に親ディレクトリを作成
        self.parent.mkdir(parents=True, exist_ok=True)
        self.mkdir(mode, parents=False, exist_ok=exist_ok)
    except OSError:
        if not exist_ok or not self.is_dir():
            raise
```

ポイントは**遅延評価的なアプローチ**を取っていること。最初から全パスを分解するのではなく、まず直接`os.mkdir()`を試み、失敗したら親を再帰的に作成します。

### `mode` パラメータの注意点

```python
path.mkdir(mode=0o755, parents=True)
```

**重要な落とし穴**: `mode`は**最終的なディレクトリにのみ適用**されます。中間ディレクトリは`mode=0o777`で作成され、その後umaskでマスクされます。

```python
# 例: umask=0o022 の環境で
Path("/tmp/a/b/c").mkdir(mode=0o700, parents=True)
# /tmp/a     → 0o755 (0o777 & ~0o022)
# /tmp/a/b   → 0o755
# /tmp/a/b/c → 0o700  ← modeが適用されるのはここだけ
```

### `exist_ok=True` との組み合わせ

```python
path = Path("/tmp/existing_dir")
path.mkdir(parents=True, exist_ok=False)  # すでに存在するとFileExistsError
path.mkdir(parents=True, exist_ok=True)   # 存在してもエラーにならない
```

## `os.makedirs()` との比較

`Path.mkdir(parents=True)`は内部的には異なるアプローチを取っています：

| 特性 | `Path.mkdir(parents=True)` | `os.makedirs()` |
|------|---------------------------|-----------------|
| 実装 | 再帰的に`os.mkdir()`を呼ぶ | 事前にパス分解してループ |
| mode適用 | 最終ディレクトリのみ | 全ディレクトリ（Python 3.7+） |
| 戻り値 | `None` | `None` |

## 典型的なユースケースとイディオム

```python
from pathlib import Path

# 設定ファイルのディレクトリを安全に作成
config_dir = Path.home() / ".config" / "myapp"
config_dir.mkdir(parents=True, exist_ok=True)

# 出力ファイルの親ディレクトリを確保
output_file = Path("results/2024/experiment_001/data.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text('{"result": "success"}')
```

## レースコンディションへの耐性

`parents=True, exist_ok=True`の組み合わせは、マルチプロセス環境でも比較的安全です。ただし、完全なアトミック性は保証されないため、高並列環境では`try/except`でラップするのが堅牢です：

```python
try:
    path.mkdir(parents=True, exist_ok=True)
except OSError as e:
    if not path.is_dir():
        raise
```

# `Path.mkdir(parents=True)` の詳細解説

## 基本シグネチャ

```python
Path.mkdir(mode=0o777, parents=False, exist_ok=False)
```

## 各パラメータの動作

### `parents=True` の効果

このフラグは、**中間ディレクトリを再帰的に作成するかどうか**を制御します。

```python
from pathlib import Path

path = Path("/tmp/a/b/c/d")

# parents=False (デフォルト): 親ディレクトリが存在しないとFileNotFoundError
path.mkdir()  # FileNotFoundError: [Errno 2] No such file or directory

# parents=True: 存在しない中間ディレクトリをすべて作成
path.mkdir(parents=True)  # /tmp/a, /tmp/a/b, /tmp/a/b/c, /tmp/a/b/c/d を順に作成
```

### 内部実装の仕組み

CPythonの実装を見ると、`parents=True`の場合は以下のような処理が行われます：

```python
# 簡略化した内部ロジック（実際のCPython実装に基づく）
def mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        os.mkdir(self, mode)
    except FileNotFoundError:
        if not parents or self.parent == self:
            raise
        # 再帰的に親ディレクトリを作成
        self.parent.mkdir(parents=True, exist_ok=True)
        self.mkdir(mode, parents=False, exist_ok=exist_ok)
    except OSError:
        if not exist_ok or not self.is_dir():
            raise
```

ポイントは**遅延評価的なアプローチ**を取っていること。最初から全パスを分解するのではなく、まず直接`os.mkdir()`を試み、失敗したら親を再帰的に作成します。

### `mode` パラメータの注意点

```python
path.mkdir(mode=0o755, parents=True)
```

**重要な落とし穴**: `mode`は**最終的なディレクトリにのみ適用**されます。中間ディレクトリは`mode=0o777`で作成され、その後umaskでマスクされます。

```python
# 例: umask=0o022 の環境で
Path("/tmp/a/b/c").mkdir(mode=0o700, parents=True)
# /tmp/a     → 0o755 (0o777 & ~0o022)
# /tmp/a/b   → 0o755
# /tmp/a/b/c → 0o700  ← modeが適用されるのはここだけ
```

### `exist_ok=True` との組み合わせ

```python
path = Path("/tmp/existing_dir")
path.mkdir(parents=True, exist_ok=False)  # すでに存在するとFileExistsError
path.mkdir(parents=True, exist_ok=True)   # 存在してもエラーにならない
```

## `os.makedirs()` との比較

`Path.mkdir(parents=True)`は内部的には異なるアプローチを取っています：

| 特性 | `Path.mkdir(parents=True)` | `os.makedirs()` |
|------|---------------------------|-----------------|
| 実装 | 再帰的に`os.mkdir()`を呼ぶ | 事前にパス分解してループ |
| mode適用 | 最終ディレクトリのみ | 全ディレクトリ（Python 3.7+） |
| 戻り値 | `None` | `None` |

## 典型的なユースケースとイディオム

```python
from pathlib import Path

# 設定ファイルのディレクトリを安全に作成
config_dir = Path.home() / ".config" / "myapp"
config_dir.mkdir(parents=True, exist_ok=True)

# 出力ファイルの親ディレクトリを確保
output_file = Path("results/2024/experiment_001/data.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text('{"result": "success"}')
```

## レースコンディションへの耐性

`parents=True, exist_ok=True`の組み合わせは、マルチプロセス環境でも比較的安全です。ただし、完全なアトミック性は保証されないため、高並列環境では`try/except`でラップするのが堅牢です：

```python
try:
    path.mkdir(parents=True, exist_ok=True)
except OSError as e:
    if not path.is_dir():
        raise
```