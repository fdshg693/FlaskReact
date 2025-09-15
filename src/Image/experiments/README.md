# Experiments Module - 画像分類訓練スクリプト

## 概要

実験的な画像分類モデル（WoodNet）の訓練を実行するメインスクリプトモジュール。CNN による木材分類タスクの学習パイプラインを提供します。

## 主要機能

- **自動化された訓練パイプライン**: 設定ファイルから自動的に訓練環境を構築
- **動的実験管理**: タイムスタンプ付き実験名で結果を自動分類
- **包括的なログ記録**: 訓練過程の詳細記録と可視化
- **チェックポイント管理**: ベストモデルの自動保存

## メイン関数

### `train_wood_classification.main()`

木材分類モデルの訓練を実行するメイン関数。

```python
def main() -> None:
    """Main training function for wood classification."""
```

**処理フロー:**

1. YAML 設定ファイル (`default_config.yaml`) の読み込み
2. データセット初期化とクラス数の自動検出
3. 訓練/検証データの分割 (8:2)
4. WoodNet モデルの初期化
5. 損失関数・最適化器の設定
6. ClassificationTrainer による訓練実行
7. 結果の自動保存

**使用するコアコンポーネント:**

- `ConfigManager`: YAML 設定管理
- `WoodDataset`: データローダー初期化
- `WoodNet`: CNN モデル構築
- `ClassificationTrainer`: 訓練ループ実行
- `Logger`: 訓練ログ記録
- `CheckpointManager`: モデル保存管理

## 実行方法

```bash
# プロジェクトルートから実行
uv run python -m src.image.experiments.train_wood_classification
```

## 出力結果

**ログディレクトリ**: `logs/[timestamp_experiment_name]/`

- `training.log`: 詳細な訓練ログ
- `metrics.csv`: エポック別メトリクス
- `config.yaml`: 使用した設定
- `confusion_matrix_*.png`: 混同行列可視化
- `training_curves.png`: 損失・精度曲線

**チェックポイントディレクトリ**: `checkpoints/[timestamp_experiment_name]/`

- `best_accuracy.pth`: 最高精度モデル
- `best_loss.pth`: 最低損失モデル
- `latest.pth`: 最新モデル

## 依存関係

訓練スクリプトは以下のコアモジュールに依存:

- `image.core.models.wood_net`: CNN アーキテクチャ
- `image.core.datasets.wood_dataset`: データ処理
- `image.core.trainers.classification_trainer`: 訓練ロジック
- `image.config.base_config`: 設定管理

## 使用先

このスクリプトで訓練されたモデルは以下で利用:

- `core.evaluation.evaluate_model`: 単体推論・一括評価
- Flask API エンドポイント: Web サービス統合
- Streamlit アプリ: インタラクティブ UI
