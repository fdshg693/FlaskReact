# 画像解析・機械学習プロジェクト 包括的ドキュメント

## 📋 概要

このプロジェクトは、PyTorch を使用した画像分類（特に木材分類）のための包括的な機械学習パイプラインです。CNN（WoodNet）による画像分類、訓練、評価、及び推論機能を提供します。

**プロジェクトの主な特徴:**

- モダンな Python アーキテクチャ（型ヒント、抽象クラス、設定管理）
- YAML 設定ファイルによる実験管理
- 詳細なログ記録と可視化
- GPU 対応の自動検出
- チェックポイント管理システム
- 包括的なデータ拡張

---

## 🏗️ プロジェクト構造

```
src/image/
├── __init__.py                          # パッケージ初期化
├── README.md                           # 基本説明（既存）
├── todo.md                             # TODO管理（空）
├── evaluate.py                         # 評価スクリプトエントリーポイント
├── create_test_dataset.py             # テストデータセット生成
│
├── config/                             # 設定管理
│   ├── __init__.py
│   ├── base_config.py                  # 設定クラス定義
│   ├── default_config.yaml             # デフォルト設定
│   └── experiment_configs/             # 実験別設定
│       └── advanced_wood_classification.yaml
│
├── core/                               # コアモジュール
│   ├── __init__.py
│   ├── datasets/                       # データセット処理
│   │   ├── __init__.py
│   │   ├── base_dataset.py            # データセット基底クラス
│   │   └── wood_dataset.py            # 木材データセット実装
│   │
│   ├── models/                         # ニューラルネットワークモデル
│   │   ├── __init__.py
│   │   ├── base_model.py              # モデル基底クラス
│   │   └── wood_net.py                # WoodNet CNN実装
│   │
│   ├── trainers/                       # 訓練ロジック
│   │   ├── __init__.py
│   │   ├── base_trainer.py            # トレーナー基底クラス
│   │   └── classification_trainer.py  # 分類訓練実装
│   │
│   ├── utils/                          # ユーティリティ
│   │   ├── __init__.py
│   │   ├── checkpoint.py              # チェックポイント管理
│   │   ├── logger.py                  # ログ管理
│   │   ├── metrics.py                 # メトリクス計算
│   │   └── visualization.py           # 可視化
│   │
│   └── evaluation/                     # モデル評価
│       ├── __init__.py
│       ├── evaluate_model.py          # CLIベース評価
│       └── evaluator.py               # 評価ロジック実装
│
├── experiments/                        # 実験スクリプト
│   ├── __init__.py
│   └── train_wood_classification.py   # メイン訓練スクリプト
│
├── dataset/                            # データセット格納
│   └── test_dataset/                   # テスト用データセット
│       ├── birch/                      # バーチ材画像
│       ├── oak/                        # オーク材画像
│       └── pine/                       # パイン材画像
│
├── logs/                               # 訓練ログ
│   └── [timestamp_experiment_name]/   # 実験別ログディレクトリ
│       ├── training.log               # 訓練ログ
│       ├── metrics.csv                # メトリクスCSV
│       ├── config.yaml                # 使用設定
│       ├── confusion_matrix_*.png     # 混同行列
│       └── training_curves.png        # 訓練曲線
│
├── checkpoints/                        # モデルチェックポイント
│   └── [timestamp_experiment_name]/   # 実験別チェックポイント
│       ├── best_accuracy.pth          # 最高精度モデル
│       ├── best_loss.pth              # 最低損失モデル
│       ├── latest.pth                 # 最新モデル
│       └── checkpoint_info.json       # チェックポイント情報
│
└── tests/                              # テストコード
    └── [test files]
```

---

## ⚙️ 主要コンポーネント詳細

### 1. 設定管理システム (`config/`)

#### `base_config.py`

```python
class BaseConfig:
    """YAML設定管理クラス"""

class ConfigManager:
    """設定ファイル管理"""
    - load_config(path) -> BaseConfig
    - save_config() -> None
    - validate_config() -> bool
```

#### `default_config.yaml`

```yaml
# データセット設定
dataset_name: "test_dataset"
dataset_path: "data/machineLearning/image/test_dataset"
img_size: 128
img_scale: 1.5

# モデル設定
layer: 3
num_hidden: 4096
dropout_rate: 0.2

# 訓練設定
epoch: 3
batch_size_train: 10
learning_rate: 0.001
```

### 2. データセット (`core/datasets/`)

#### `wood_dataset.py`

```python
class WoodDataset(BaseDataset):
    """木材分類用データセット"""

    def __init__(self, dataset_path, num_class, img_size, ...):
        # データ拡張設定
        - scale: スケール変換
        - brightness: 明度変換
        - rotation: 回転変換
        - flip: 反転変換

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        # 画像とラベルのペアを返す

    def get_class_weights(self) -> torch.Tensor:
        # クラス不均衡対応の重み計算
```

**対応する入力形式:**

- 画像フォーマット: JPG, PNG, BMP
- ディレクトリ構造: `dataset_name/class_name/image_files`
- 画像サイズ: 自動リサイズ（設定可能）

### 3. モデルアーキテクチャ (`core/models/`)

#### `wood_net.py`

```python
class WoodNet(BaseModel):
    """CNN画像分類モデル"""

    def __init__(self, num_class, img_size, layer, num_hidden, ...):
        # 動的CNN構造
        - 設定可能な層数
        - Mishアクティベーション
        - L2正規化Softmax
        - ドロップアウト対応

    def forward(self, x) -> torch.Tensor:
        # 順伝播処理
```

**モデル構造:**

```
Input(3, img_size, img_size)
    ↓
Conv2d(3 → 100) + Mish + Stride(2)
    ↓
Conv2d(100 → 144) + Mish + Stride(2)  # layer=1の場合
    ↓
... (layer数分繰り返し)
    ↓
Flatten + Dropout
    ↓
Linear(→ num_hidden) + Mish
    ↓
Linear(num_hidden → num_class)
    ↓
L2Softmax (optional)
```

**モデル構造の詳細説明:**

1. **Input(3, img_size, img_size)**

   - 入力データの形状を表す
   - `3`: カラー画像のチャネル数（RGB）
   - `img_size`: 画像の縦・横サイズ（例: 128×128）
   - 例: (3, 128, 128) = RGB 3 チャネル、128×128 ピクセルの画像

2. **Conv2d(3 → 100) + Mish + Stride(2)**

   - `Conv2d`: 2 次元畳み込み層
   - `3 → 100`: 入力チャネル数 3 から出力チャネル数 100 に変換
   - `Mish`: 活性化関数（ReLU より性能が良い）
   - `Stride(2)`: 2 ピクセルずつスライドするため、画像サイズが半分になる
   - 画像サイズ変化: 128×128 → 64×64

3. **Conv2d(100 → 144) + Mish + Stride(2)**

   - チャネル数を 100 から 144 に増加（特徴量をより多く抽出）
   - 計算式: (10 + 1\*2)² = 12² = 144
   - 画像サイズ変化: 64×64 → 32×32

4. **... (layer 数分繰り返し)**

   - 設定した`layer`パラメータ分だけ畳み込み層を追加
   - 各層でチャネル数が増加: (10 + i\*2)² の計算式
   - layer=3 の場合: 100 → 144 → 196 → 256 チャネル
   - 画像サイズは各層で半分に: 32×32 → 16×16 → 8×8

5. **Flatten + Dropout**

   - `Flatten`: 2 次元の特徴マップを 1 次元ベクトルに変換
   - 例: (256, 8, 8) → (16384,) の 1 次元ベクトル
   - `Dropout`: 過学習防止のため、一部のニューロンをランダムに無効化

6. **Linear(→ num_hidden) + Mish**

   - `Linear`: 全結合層（線形変換）
   - 特徴ベクトルを`num_hidden`個のユニットに変換（例: 4096 個）
   - `Mish`: 活性化関数を適用

7. **Linear(num_hidden → num_class)**

   - 最終出力層
   - `num_hidden`から`num_class`（クラス数）に変換
   - 例: 4096 → 3（oak, pine, birch の 3 クラス）

8. **L2Softmax (optional)**
   - `L2正規化`: 特徴ベクトルの長さを 1 に正規化
   - `Softmax`: 各クラスの確率に変換（合計が 1 になる）
   - オプション機能（`l2softmax=True`の場合のみ適用）

### 4. 訓練システム (`core/trainers/`)

#### `classification_trainer.py`

```python
class ClassificationTrainer(BaseTrainer):
    """分類訓練管理クラス"""

    def fit(self, train_loader, val_loader, epochs):
        # 訓練ループ実行
        - 早期停止対応
        - 学習率スケジューリング
        - チェックポイント自動保存
        - メトリクス記録

    def _train_epoch(self) -> Dict[str, float]:
        # 1エポック訓練

    def _validate_epoch(self) -> Dict[str, float]:
        # 1エポック検証
```

### 5. 評価システム (`core/evaluation/`)

#### `evaluator.py`

```python
class ModelEvaluator:
    """モデル評価クラス"""

    def predict_image(self, image_path) -> Dict:
        # 単一画像の推論

    def evaluate_directory(self, dir_path) -> Dict:
        # ディレクトリ一括評価

    def visualize_prediction(self, image_path, save_path):
        # 予測結果可視化
```

### 6. ユーティリティ (`core/utils/`)

#### `logger.py`

```python
class Logger:
    """訓練ログ管理"""
    - log_info(message)
    - log_metrics(epoch, metrics)
    - log_config(config_dict)
    - save_metrics_csv()
```

#### `visualization.py`

```python
class Visualizer:
    """可視化機能"""
    - plot_training_curves(metrics)
    - plot_confusion_matrix(cm, class_names)
    - save_sample_predictions()
```

#### `checkpoint.py`

```python
class CheckpointManager:
    """チェックポイント管理"""
    - save_checkpoint(model, optimizer, epoch, metrics)
    - load_checkpoint(path) -> Dict
    - save_best_models(metric_name, metric_value)
```

#### `metrics.py`

```python
class MetricsCalculator:
    """メトリクス計算"""
    - update(predictions, targets)
    - get_accuracy() -> float
    - get_confusion_matrix() -> np.ndarray
    - get_classification_report() -> str
```

---

## 🚀 使用方法

### 1. セットアップ

```bash
# プロジェクトルートに移動
cd FlaskReact

# Python環境設定（プロジェクトの指示に従う）
uv sync

# テストデータセット作成（初回のみ）
uv run python -m src.image.create_test_dataset
```

### 2. 訓練実行

#### 基本的な訓練

```bash
# デフォルト設定で訓練実行
uv run python -m src.image.experiments.train_wood_classification
```

#### カスタム設定での訓練

```yaml
# config/experiment_configs/my_experiment.yaml
dataset_name: "my_dataset"
img_size: 256
layer: 4
num_hidden: 8192
epoch: 100
batch_size_train: 32
```

```bash
# カスタム設定で実行（設定ファイルパス変更が必要）
```

### 3. モデル評価

#### 利用可能なチェックポイント確認

```bash
uv run python -m src.image.core.evaluation.evaluate_model --list-checkpoints
```

#### 単一画像の推論

```bash
uv run python -m src.image.core.evaluation.evaluate_model \
    --model "./checkpoints/[experiment_name]/best_accuracy.pth" \
    --image "./dataset/test_dataset/birch/image_01.jpg" \
    --visualize
```

#### ディレクトリ一括評価

```bash
uv run python -m src.image.core.evaluation.evaluate_model \
    --model "./checkpoints/[experiment_name]/best_accuracy.pth" \
    --directory "./dataset/test_dataset/" \
    --output "./results.json"
```

#### Visual Studio Code デバッガーでの実行

1. `src/image/core/evaluation/evaluate_model.py`を開く
2. "Python Debugger: Current File with Arguments (src layout)"を選択
3. 引数設定:

```
--model "./checkpoints/[experiment_name]/best_accuracy.pth"
--image "./dataset/test_dataset/birch/image_01.jpg"
```

4. F5 で実行

### 4. テストデータセット作成

```bash
# テスト用ダミー画像生成
uv run python -m src.image.create_test_dataset

# 生成される構造:
# data/machineLearning/image/test_dataset/
# ├── birch/    (バーチ材 - ベージュ系)
# ├── oak/      (オーク材 - 茶色系)
# └── pine/     (パイン材 - 緑系)
```

---

## 📊 入力・出力形式

### 入力形式

#### 1. データセット構造

```
your_dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

#### 2. 設定ファイル (YAML)

```yaml
# 必須設定
dataset_name: "データセット名"
dataset_path: "データセットへのパス"
img_size: 128 # 画像サイズ
layer: 3 # CNN層数
num_hidden: 4096 # 隠れ層ユニット数

# オプション設定
dropout_rate: 0.2 # ドロップアウト率
learning_rate: 0.001 # 学習率
batch_size_train: 16 # バッチサイズ
epoch: 100 # エポック数

# データ拡張
img_scale: 1.5 # スケーリング係数
aug_brightness: 50.0 # 明度変化範囲
aug_scale: 0.5 # スケール拡張範囲
```

#### 3. 評価コマンドライン引数

```bash
# 必須引数
--model PATH                    # モデルファイル(.pth)
--image PATH | --directory PATH # 評価対象

# オプション引数
--output PATH                   # 結果保存先(JSON)
--visualize                     # 可視化表示
--device cuda|cpu               # 実行デバイス
--img-size SIZE                 # 画像サイズ
```

### 出力形式

#### 1. 訓練ログ構造

```
logs/[timestamp_experiment_name]/
├── training.log                # テキストログ
├── metrics.csv                 # メトリクスCSV
├── config.yaml                 # 使用設定
├── confusion_matrix_epoch_*.png # 混同行列(エポック毎)
├── training_curves.png         # 訓練曲線
└── summary.json                # 実験サマリー
```

#### 2. チェックポイント構造

```
checkpoints/[timestamp_experiment_name]/
├── best_accuracy.pth           # 最高精度モデル
├── best_loss.pth               # 最低損失モデル
├── latest.pth                  # 最新モデル
├── checkpoint_epoch_*.pth      # エポック別チェックポイント
└── checkpoint_info.json        # チェックポイント情報
```

#### 3. 評価結果 (JSON)

```json
{
  "predicted_class": 0,
  "confidence": 0.8945,
  "probabilities": [0.8945, 0.0823, 0.0232],
  "image_path": "./dataset/test_dataset/birch/image_01.jpg",
  "model_path": "./checkpoints/.../best_accuracy.pth",
  "processing_time": 0.0234
}
```

#### 4. 一括評価結果 (JSON)

```json
{
  "total_images": 30,
  "successful_predictions": 28,
  "failed_predictions": 2,
  "average_confidence": 0.7823,
  "min_confidence": 0.4567,
  "max_confidence": 0.9876,
  "class_distribution": {
    "0": 10,
    "1": 9,
    "2": 9
  },
  "detailed_results": [...]
}
```

#### 5. メトリクス CSV

```csv
epoch,train_loss,train_acc,val_loss,val_acc,learning_rate
0,1.2345,0.3333,1.1234,0.4000,0.001
1,0.9876,0.6000,0.8765,0.7000,0.001
...
```

---

## 🔗 関数・クラス参照関係

### 主要な呼び出し関係

#### 1. 訓練フロー

```python
train_wood_classification.main()
    ├── ConfigManager.get_config()
    ├── WoodDataset.__init__()
    ├── WoodNet.__init__()
    ├── ClassificationTrainer.__init__()
    └── ClassificationTrainer.fit()
        ├── ClassificationTrainer._train_epoch()
        ├── ClassificationTrainer._validate_epoch()
        ├── CheckpointManager.save_checkpoint()
        ├── Logger.log_metrics()
        └── Visualizer.plot_confusion_matrix()
```

#### 2. 評価フロー

```python
evaluate_model.main()
    ├── ModelEvaluator.__init__()
    │   ├── WoodNet.__init__()
    │   └── CheckpointManager.load_checkpoint()
    ├── ModelEvaluator.predict_image()
    │   ├── WoodDataset._preprocess_image()
    │   └── WoodNet.forward()
    └── ModelEvaluator.visualize_prediction()
```

#### 3. データ処理フロー

```python
WoodDataset.__getitem__()
    ├── cv2.imread()                    # 画像読み込み
    ├── WoodDataset._apply_augmentation() # データ拡張
    ├── cv2.resize()                    # リサイズ
    └── torch.tensor()                  # テンソル変換
```

### 外部ライブラリ依存関係

#### PyTorch 関連

```python
# core/models/wood_net.py
import torch
import torch.nn as nn

# core/trainers/classification_trainer.py
import torch.optim as optim
from torch.utils.data import DataLoader
```

#### 画像処理

```python
# core/datasets/wood_dataset.py
import cv2                    # OpenCV
from PIL import Image        # Pillow

# core/utils/visualization.py
import matplotlib.pyplot as plt
```

#### 科学計算・機械学習

```python
# experiments/train_wood_classification.py
from sklearn.model_selection import train_test_split

# core/utils/metrics.py
from sklearn.metrics import classification_report
import numpy as np
```

#### 設定・ログ管理

```python
# config/base_config.py
import yaml

# core/utils/logger.py
import logging
import csv
import json
```

---

## 🔧 カスタマイズ・拡張方法

### 1. 新しいモデルの追加

```python
# core/models/my_model.py
from .base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_class, **kwargs):
        super().__init__(num_class, **kwargs)
        # カスタムアーキテクチャ定義

    def forward(self, x):
        # 順伝播処理
        return x
```

### 2. カスタムデータセットの追加

```python
# core/datasets/my_dataset.py
from .base_dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path, **kwargs)
        # カスタムデータ処理

    def __getitem__(self, index):
        # カスタムデータローディング
        return image, label
```

### 3. 新しいメトリクスの追加

```python
# core/utils/metrics.py
class MetricsCalculator:
    def get_custom_metric(self):
        # カスタムメトリクス計算
        return custom_value
```

### 4. 実験設定テンプレートの追加

```yaml
# config/experiment_configs/my_experiment.yaml
# 基本設定を継承
dataset_name: "my_custom_dataset"
img_size: 224
layer: 5
num_hidden: 8192

# カスタム設定
custom_param1: value1
custom_param2: value2
```

---

## 🐛 トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA 関連エラー

```bash
# エラー: CUDA out of memory
# 解決: バッチサイズを減らす
batch_size_train: 8  # 16から8に変更
```

#### 2. データセット読み込みエラー

```python
# エラー: No such file or directory
# 解決: パスの確認
dataset_path: "data/machineLearning/image/test_dataset"  # 絶対パス推奨
```

#### 3. モデル読み込みエラー

```python
# エラー: checkpoint file not found
# 解決: チェックポイント確認
uv run python -m src.image.core.evaluation.evaluate_model --list-checkpoints
```

#### 4. 設定ファイルエラー

```yaml
# エラー: Invalid YAML syntax
# 解決: インデントと型の確認
epoch: 100 # 文字列ではなく数値
dropout_rate: 0.2 # 小数点形式
```

### デバッグ方法

#### 1. ログレベル変更

```python
# core/utils/logger.py
logger.setLevel(logging.DEBUG)  # より詳細なログ
```

#### 2. チェックポイント確認

```python
import torch
checkpoint = torch.load("./checkpoints/.../best_accuracy.pth")
print(checkpoint.keys())  # 保存内容確認
```

#### 3. データセット確認

```python
from image.core.datasets.wood_dataset import WoodDataset
dataset = WoodDataset("./dataset/test_dataset")
print(f"Classes: {dataset.get_num_class()}")
print(f"Samples: {len(dataset)}")
```

---

## 📈 性能最適化

### 1. 訓練高速化

- GPU 使用: `device: "cuda"`
- バッチサイズ最適化: メモリに応じて調整
- データローダー並列化: `num_workers > 0`

### 2. メモリ使用量削減

- 画像サイズ縮小: `img_size: 64`
- バッチサイズ削減: `batch_size_train: 8`
- 混合精度訓練（実装予定）

### 3. 精度向上

- データ拡張強化: `aug_brightness`, `aug_scale`調整
- モデル深化: `layer`数増加
- アンサンブル学習（実装予定）

---

## 🧪 テスト

### テスト実行方法

```bash
# 全テスト実行
pytest src/image/tests/

# 特定テスト実行
pytest src/image/tests/test_models.py
pytest src/image/tests/test_datasets.py

# カバレッジ付き実行
pytest --cov=src/image src/image/tests/
```

### テスト項目

- モデル構造の正常性
- データセット読み込み機能
- 設定ファイル解析
- チェックポイント保存/読み込み
- メトリクス計算精度

---

## 📝 更新履歴・今後の予定

### 実装済み機能

- ✅ CNN 画像分類（WoodNet）
- ✅ YAML 設定管理システム
- ✅ チェックポイント管理
- ✅ 包括的なログ記録
- ✅ 混同行列・訓練曲線可視化
- ✅ CLI 評価インターフェース
- ✅ データ拡張機能
- ✅ クラス不均衡対応

### 今後の予定

- 🔄 混合精度訓練対応
- 🔄 転移学習機能
- 🔄 アンサンブル学習
- 🔄 モデル圧縮・量子化
- 🔄 リアルタイム推論 API
- 🔄 Web UI 統合

---

## 📞 サポート・連絡先

プロジェクト関連の質問や問題については、以下のリソースを参照してください：

- **プロジェクト README**: `src/image/README.md`
- **設定例**: `config/default_config.yaml`
- **API 文書**: 各クラスの docstring 参照
- **テストコード**: `tests/`ディレクトリ

---

_この文書は src/image/ プロジェクトの包括的な技術文書です。_
_最終更新: 2025 年 9 月 15 日_
