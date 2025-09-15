# Utils モジュール

画像分類プロジェクトの訓練・評価に必要なユーティリティ機能を提供するモジュールです。

## 主要機能

### 1. CheckpointManager (checkpoint.py)

**概要**: モデルのチェックポイント保存・読み込み管理

**主要メソッド**:

- `save_checkpoint()`: 通常・最高精度・最低損失モデルの自動保存
- `load_checkpoint()`: 保存済みチェックポイントの読み込み
- `list_checkpoints()`: 利用可能なチェックポイント一覧

**使用例**:

```python
manager = CheckpointManager("./checkpoints/experiment_name")
manager.save_checkpoint(model, optimizer, epoch, loss, accuracy, is_best_acc=True)
checkpoint = manager.load_checkpoint("./checkpoints/best_accuracy.pth")
```

**使用先**: `ClassificationTrainer`, `ModelEvaluator`

### 2. Logger (logger.py)

**概要**: 訓練プロセスの包括的ログ記録

**主要メソッド**:

- `log_info()`: 情報ログの出力
- `log_metrics()`: エポック毎のメトリクス記録
- `save_metrics_csv()`: CSV 形式でのメトリクス保存

**使用例**:

```python
logger = Logger("./logs/experiment_name")
logger.log_info("Starting training...")
logger.log_metrics(epoch=1, metrics={"loss": 0.5, "accuracy": 0.8})
```

**使用先**: `train_wood_classification.py`, `ClassificationTrainer`

### 3. MetricsCalculator (metrics.py)

**概要**: 分類精度、損失値、混同行列等の計算

**主要メソッド**:

- `update()`: バッチ毎の予測結果でメトリクス更新
- `get_accuracy()`: 累積精度の取得
- `get_confusion_matrix()`: 混同行列の生成

**使用例**:

```python
metrics = MetricsCalculator(num_classes=3)
metrics.update(predictions, targets, loss)
accuracy = metrics.get_accuracy()
```

**使用先**: `ClassificationTrainer`, `ModelEvaluator`

### 4. Visualizer (visualization.py)

**概要**: 訓練曲線、混同行列、予測結果の可視化

**主要メソッド**:

- `plot_training_curves()`: 損失・精度の訓練曲線プロット
- `plot_confusion_matrix()`: 混同行列のヒートマップ生成
- `save_sample_predictions()`: 予測結果のサンプル画像保存

**使用例**:

```python
visualizer = Visualizer("./logs/experiment_name")
visualizer.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
visualizer.plot_confusion_matrix(confusion_matrix, class_names)
```

**使用先**: `ClassificationTrainer`, `ModelEvaluator`
