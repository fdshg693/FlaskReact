/_ コメント
Base dataset と Wood dataset の役割の記述があってもいいかも
_/

# Config モジュール

画像解析・機械学習プロジェクトの設定管理システムです。YAML ファイルベースの設定管理により、実験の管理と再現性を提供します。

## 概要

本モジュールは、機械学習実験の設定を統一的に管理するための仕組みを提供します。ベース設定と実験固有の設定を分離し、設定の継承・上書き機能により柔軟な実験管理を実現します。

## 主要クラス・関数

### BaseConfig クラス

YAML 設定ファイルからの設定読み込みと管理を行うクラスです。

```python
from src.image.config.base_config import BaseConfig

# YAML ファイルから設定読み込み
config = BaseConfig.from_yaml('config.yaml')

# デフォルト設定のみ使用
config = BaseConfig.from_default()

# 設定を辞書として取得
config_dict = config.to_dict()

# 設定を YAML ファイルに保存
config.save_yaml('output_config.yaml')
```

### ConfigManager クラス

設定の読み込み、更新、保存を統合的に管理するクラスです。

```python
from src.image.config.base_config import ConfigManager

# 設定マネージャー初期化
manager = ConfigManager('experiment.yaml')

# 現在の設定取得
config = manager.get_config()

# 設定の動的更新
manager.update_config(
    learning_rate=0.001,
    batch_size_train=16
)

# 設定保存
manager.save_config()

# 特定の実験設定読み込み
manager.load_experiment_config('advanced_wood_classification')
```

## 設定ファイル構造

- `default_config.yaml`: ベース設定（全実験共通のデフォルト値）
- `experiment_configs/`: 実験固有の設定ファイル格納ディレクトリ

実験設定は、ベース設定を継承し、必要な項目のみ上書きする構造です。

## 使用先・活用場面

### 訓練スクリプト (`experiments/train_wood_classification.py`)

```python
# 設定読み込み
config_manager = ConfigManager(config_path=yaml_config_path)
config = config_manager.get_config()

# 各種設定値の利用
device = config.device
batch_size = config.batch_size_train
learning_rate = config.learning_rate
```

### モデル評価 (`core/evaluation/evaluator.py`)

```python
# チェックポイントから設定復元
model_config = checkpoint['config']
num_classes = model_config.get("num_class", 3)
img_size = model_config.get("img_size", 128)
```

### ログ・チェックポイント管理

- **Logger**: 実験名とログディレクトリの設定
- **CheckpointManager**: チェックポイント保存先の設定
- **Visualizer**: 可視化結果の保存先設定

## 自動機能

- **実験名生成**: タイムスタンプと設定パラメータから自動生成
- **CUDA 検出**: GPU 環境の自動検出と device 設定
- **パス管理**: ログ・チェックポイントディレクトリの自動設定
