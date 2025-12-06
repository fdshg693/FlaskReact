# 木材画像分類プロジェクト

PyTorch を使用した木材画像の自動分類システムです。CNN ベースのニューラルネットワーク（WoodNet）により、木材の種類を高精度で分類します。

## 🚀 特徴

- **モダンな Python アーキテクチャ**: 設定管理、ロギング、チェックポイント管理を含む完全な機械学習パイプライン
- **柔軟な設定システム**: YAML ベースの設定管理で実験の再現性を確保
- **包括的なデータ拡張**: 反転、輝度変更、スケール変更によるロバストな学習
- **詳細なログと可視化**: 学習進捗の追跡と混同行列による評価
- **GPU 対応**: CUDA 利用時の自動 GPU 使用
- **早期停止とスケジューラー**: 過学習防止とリソース効率化

## 📁 プロジェクト構造

```
├── src/                       # メインソースコード
│   ├── models/               # ニューラルネットワークモデル
│   │   ├── base_model.py     # ベースモデルクラス
│   │   └── wood_net.py       # WoodNet（カスタムCNN）
│   ├── datasets/             # データ処理
│   │   ├── base_dataset.py   # ベースデータセットクラス
│   │   └── wood_dataset.py   # 木材データセット
│   ├── trainers/             # 学習管理
│   │   ├── base_trainer.py   # ベーストレーナークラス
│   │   └── classification_trainer.py # 分類学習トレーナー
│   ├── utils/                # ユーティリティ
│   │   ├── checkpoint.py     # チェックポイント管理
│   │   ├── logger.py         # 構造化ログ管理
│   │   ├── metrics.py        # メトリクス計算
│   │   └── visualization.py  # 学習曲線・混同行列可視化
│   └── evaluation/           # モデル評価
├── config/                   # 設定管理
│   ├── base_config.py        # 設定管理クラス
│   ├── default_config.yaml   # デフォルト設定
│   └── experiment_configs/   # 実験別設定
├── experiments/              # 実験スクリプト
│   └── train_wood_classification.py # メイン学習スクリプト
├── dataset/                  # データセット置き場
├── logs/                     # 学習ログ
├── checkpoints/              # モデルチェックポイント
├── trimming/                 # 画像前処理ツール
└── tests/                    # テストコード
```

### 主要パッケージ

- **PyTorch**: ディープラーニングフレームワーク
- **torchvision**: 画像処理
- **OpenCV**: 画像前処理
- **NumPy**: 数値計算
- **Matplotlib**: 可視化
- **scikit-learn**: メトリクス計算
- **PyYAML**: 設定ファイル管理
- **pytest**: テストフレームワーク

## 🎯 使用方法

### 1. データセット準備

画像データを以下の構造で配置：

```
dataset/
└── [dataset_name]/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

### 2. 設定調整

`config/default_config.yaml`で設定を調整：

```yaml
# データセット設定
dataset_name: "your_dataset"
dataset_path: "./dataset/your_dataset/"
img_size: 128
img_scale: 1.5

# モデル設定
layer: 3
num_hidden: 4096
dropout_rate: 0.2

# 学習設定
epoch: 20
batch_size_train: 16
learning_rate: 0.001
```

### 3. 学習実行

```bash
python experiments/train_wood_classification.py
```

### 4. 学習済みモデルの実行

Python Debugger: Current File with Arguments (src layout)を選択した状態でsrc/image/core/evaluation/evaluate_model.pyを開いて、F5実行。
次の引数を渡す
```bash
--model "./checkpoints/2025_08_24_20_09_13_img128_layer3_hidden4096_0class_dropout0.2_scale1.5_test_dataset/best_accuracy.pth" 
--image "./dataset/test_dataset/birch/image_01.jpg"
```

## 📊 出力

学習実行時に以下が生成されます：

### ログディレクトリ: `logs/[timestamp]_[config]/`

- **学習ログ**: 損失・精度の推移
- **学習曲線**: `accuracy.png`, `loss.png`
- **混同行列**: `confusion_matrix.png`
- **設定ファイル**: 実験の再現用

### チェックポイント: `checkpoints/[timestamp]_[config]/`

- **best_accuracy.pth**: 最高精度モデル
- **best_loss.pth**: 最低損失モデル
- **latest.pth**: 最新モデル
- **checkpoint_info.json**: チェックポイント情報

## 🔧 コアコンポーネント

### WoodNet (`src/models/wood_net.py`)

- **動的 CNN 構造**: 設定可能な層数と隠れ層サイズ
- **Mish アクティベーション**: ReLU より高性能な活性化関数
- **L2 正規化ソフトマックス**: 特徴量の正規化
- **ドロップアウト**: 過学習防止

### ClassificationTrainer (`src/trainers/classification_trainer.py`)

- **学習・検証ループ**: 自動化された学習プロセス
- **早期停止**: 過学習の自動検出
- **学習率スケジューラー**: 適応的学習率調整
- **チェックポイント管理**: 定期的なモデル保存

### 設定管理 (`config/base_config.py`)

- **YAML 設定**: 人間可読な設定ファイル
- **環境変数サポート**: 本番環境での設定上書き
- **検証機能**: 設定値の妥当性チェック

## 🔨 画像前処理ツール

### `trimming/trimmer.py`

指定割合での画像端トリミング

### `trimming/hand_trim.py`

マウス操作による手動切り出し

### `trimming/image_trimmer.py`

バッチ処理用画像トリミング

## 🧪 テスト

```bash
# 全テスト実行
pytest

# 特定テスト実行
pytest tests/test_models.py
pytest tests/test_datasets.py

# 詳細出力
pytest -v

# カバレッジ付き実行（要coverage インストール）
pytest --cov=src
```

## 📈 パフォーマンス

- **GPU 使用時**: 大幅な学習高速化
- **クラス不均衡対応**: 自動重み計算
- **メモリ効率**: バッチサイズ自動調整
- **早期停止**: 無駄な学習時間削減

## 🤝 開発

### コード品質

- **型ヒント**: 全体的な型安全性
- **ドキュメント**: 包括的な docstring
- **テスト**: ユニット・統合テスト
- **リンター**: code-quality tools

### 拡張方法

1. **新しいモデル**: `src/models/`に追加
2. **カスタムデータセット**: `src/datasets/`に追加
3. **評価メトリクス**: `src/evaluation/`に追加
4. **前処理**: `src/utils/`に追加



