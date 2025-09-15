# Datasets Module

## 概要

画像分類（特に木材分類）のためのデータセット処理モジュールです。PyTorch の Dataset インターフェースを実装し、画像の読み込み、前処理、データ拡張機能を提供します。

## クラス構成

### BaseDataset（基底クラス）

- **役割**: 全データセットの共通インターフェース定義
- **実装**: 抽象基底クラス（ABC）で PyTorch の Dataset を継承
- **必須実装メソッド**: `__len__()`, `__getitem__()`

### WoodDataset（実装クラス）

- **役割**: 木材分類用の具体的なデータセット実装
- **機能**: 画像読み込み、自動ラベリング、データ拡張、クラス重み計算

## 主要機能・メソッド

### WoodDataset 初期化

```python
dataset = WoodDataset(
    dataset_path="./dataset/test_dataset",
    num_class=3,                # クラス数（0で自動検出）
    img_size=128,              # 画像サイズ
    img_scale=1.5,             # スケーリング係数
    augmentation=True,         # データ拡張有効化
    scale=0.5,                 # スケール拡張範囲
    hflip=True,                # 水平反転
    brightness=50.0            # 明度変更範囲
)
```

### データアクセス

- `__getitem__(idx)`: インデックスによる画像・ラベル取得
- `__len__()`: データセットサイズ取得
- `get_class_weights()`: クラス不均衡対応の重み取得
- `set_augmentation(bool)`: データ拡張のオン/オフ切り替え

### 内部処理メソッド

- `_scale_image()`: 画像スケーリング処理
- `_apply_augmentations()`: データ拡張適用
- `_random_crop()`: ランダムクロッピング
- `_calculate_class_weights()`: クラス重み自動計算

## 使用先

### 訓練時の使用

- **場所**: `experiments/train_wood_classification.py`
- **用途**: 訓練・検証データローダーの作成
- **組み合わせ**: `torch.utils.data.DataLoader`でバッチ処理

### 評価時の使用

- **場所**: `evaluation/evaluator.py`
- **用途**: 推論時の画像前処理
- **機能**: `_preprocess_image()`メソッドで単一画像処理

## データ形式

### 入力ディレクトリ構造

```
dataset_name/
├── class1/
│   ├── image1.jpg
│   └── image2.png
└── class2/
    └── image3.jpg
```

### 出力形式

- **画像**: torch.Tensor (C, H, W) 形式、値域[0,1]
- **ラベル**: torch.Tensor（整数ラベル）
- **サポート形式**: JPG, PNG, BMP, TIFF
