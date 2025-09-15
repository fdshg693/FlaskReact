# Trainers モジュール

画像分類モデルの訓練を管理するモジュールです。抽象基底クラスと具体的な実装クラスを提供し、統一されたインターフェースで機械学習モデルの訓練を実行します。

## 概要

trainers モジュールは、PyTorch ベースの機械学習モデルの訓練プロセスを管理します。訓練・検証・チェックポイント保存・可視化・ログ記録などの機能を統合し、効率的で再現可能な実験環境を提供します。

## 主要クラス

### BaseTrainer（抽象基底クラス）

すべてのトレーナーの基底となる抽象クラスです。共通の状態管理とインターフェースを定義します。

```python
from image.core.trainers.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    def __init__(self, model, device="cuda"):
        super().__init__(model, device)
```

**主要メソッド:**

- `train_epoch(train_loader)`: 1 エポックの訓練（抽象メソッド）
- `validate_epoch(val_loader)`: 1 エポックの検証（抽象メソッド）
- `fit(train_loader, val_loader, epochs)`: 複数エポックの訓練（抽象メソッド）
- `get_learning_rate()`: 現在の学習率取得
- `get_training_history()`: 訓練履歴取得

### ClassificationTrainer（分類タスク用実装）

画像分類タスクに特化した具体的なトレーナー実装です。CNN（WoodNet）による木材分類などに使用されます。

```python
from image.core.trainers.classification_trainer import ClassificationTrainer
import torch.optim as optim
import torch.nn as nn

# トレーナー初期化
trainer = ClassificationTrainer(
    model=wood_net,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(wood_net.parameters(), lr=0.001),
    device="cuda",
    logger=logger,
    visualizer=visualizer,
    checkpoint_manager=checkpoint_manager,
    early_stopping_patience=10
)

# 訓練実行
history = trainer.fit(train_loader, val_loader, epochs=100)
```

**高度な機能:**

- 早期停止（Early Stopping）
- 学習率スケジューリング
- 勾配クリッピング
- 自動チェックポイント保存
- リアルタイム可視化（混同行列・訓練曲線）
- データ拡張の動的制御

## 使用パターン

### 基本的な訓練フロー

```python
# 1. コンポーネント準備
model = WoodNet(num_class=3, img_size=128, layer=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. トレーナー作成
trainer = ClassificationTrainer(model, criterion, optimizer)

# 3. 訓練実行
history = trainer.fit(train_loader, val_loader, epochs=50)

# 4. 結果取得
final_accuracy = history["val_accuracies"][-1]
```

### 統合システムでの使用

プロジェクトの実験スクリプト `train_wood_classification.py` で以下のように使用されています：

```python
# experiments/train_wood_classification.py
trainer = ClassificationTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    logger=logger,
    visualizer=visualizer,
    checkpoint_manager=checkpoint_manager
)

# データセットと連携した訓練
history = trainer.fit(
    train_loader,
    val_loader,
    epochs=config.epoch,
    dataset_for_augmentation=train_dataset  # データ拡張制御
)
```

## 連携コンポーネント

- **Logger**: 詳細な訓練ログとメトリクス記録
- **Visualizer**: 混同行列・訓練曲線の可視化
- **CheckpointManager**: モデルの自動保存・復元
- **MetricsCalculator**: 精度・損失などの計算
- **WoodDataset**: データ拡張の動的制御

## 出力

訓練完了時に以下の成果物が生成されます：

- チェックポイントファイル（best_accuracy.pth, best_loss.pth）
- 訓練ログ（training.log, metrics.csv）
- 可視化画像（confusion*matrix*\*.png, training_curves.png）
- 実験サマリー（summary.json）

Trainers モジュールはプロジェクトの ML 訓練パイプラインの中核として、再現可能で高品質な機械学習実験を支援します。
