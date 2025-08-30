# Tests for Datasets Module

このディレクトリには、datasets モジュールのテストコードが含まれています。

## テストファイル

- `test_datasets.py`: BaseDataset と WoodDataset クラスの包括的なテスト

## テストの実行方法

### 全てのテストを実行

```bash
# プロジェクトルートから実行
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py -v
```

### 特定のテストクラスを実行

```bash
# BaseDataset のテストのみ実行
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py::TestBaseDataset -v

# WoodDataset のテストのみ実行
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py::TestWoodDataset -v

# 統合テストのみ実行
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py::TestDatasetIntegration -v
```

### 特定のテスト関数を実行

```bash
# 初期化テストのみ実行
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py::TestBaseDataset::test_init -v
```

### カバレッジレポートと共に実行

```bash
C:/Users/sksrd/Programing/Python/program/.venv/Scripts/python.exe -m pytest tests/test_datasets.py --cov=src.datasets --cov-report=html
```

## テストの内容

### TestBaseDataset

- 抽象基底クラス BaseDataset の基本機能をテスト
- 初期化、プロパティ取得、データセット情報の取得をテスト

### TestWoodDataset

- WoodDataset クラスの完全なテスト
- ファイル読み込み、データ拡張、画像処理機能をテスト
- 一時的なテストデータセットを作成してテスト

### TestDatasetIntegration

- 実際の使用シナリオに近い統合テスト
- PyTorch DataLoader との互換性テスト
- クラス重み計算の検証

## テストで作成される一時ファイル

テストは一時ディレクトリを作成してテスト用の画像ファイルを生成します。
テスト終了後に自動的にクリーンアップされます。

## 依存関係

テストには以下のパッケージが必要です：

- pytest
- pytest-cov
- torch
- opencv-python (cv2)
- numpy

これらは requirements.txt に含まれています。
