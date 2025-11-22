# 機械学習モジュール (src/machineLearning)

PyTorchベースの機械学習パイプライン実装。分類・回帰タスクの訓練、評価、推論、およびモデル永続化機能を提供します。

## 目的と概要

- 機械学習タスク（分類・回帰）の統一的なパイプライン提供
- データセット変換の標準化（CSV、pandas DataFrame、sklearn Bunch形式対応）
- PyTorchニューラルネットワークの訓練・評価・推論
- 学習済みモデルとスケーラーの保存・読み込み
- 学習曲線やメトリクスのログ出力（CSV、PNG）
- Flask APIエンドポイントとの連携

データフロー: `data/machineLearning/` → データセット変換 → 学習パイプライン → モデル・ログ保存 (`outputs/machineLearning/`) → Flask API経由の推論

## 主な機能

### データセット変換（dataset.py）

- **MLCompatibleDataset**: 標準データセット形式。2次元特徴量配列と1次元ターゲット配列をPydanticでバリデーション
- **MLDatasetConverter**: pandas DataFrame、sklearn Bunch、CSVファイルを統一形式に自動変換

### 統合パイプライン（pipeline.py）

- **execute_machine_learning_pipeline**: 分類/回帰の自動判定、データ分割、スケーリング、訓練を実行
- **train_and_save_pipeline**: 訓練とログ保存を一体化した便利関数

### モデルアーキテクチャ（models/）

- **BaseMLModel**: 共通前処理（データ分割、スケーリング、テンソル変換）の基底クラス
- **ClassificationMLModel**: 分類タスク用（CrossEntropyLoss、accuracy評価）
- **RegressionMLModel**: 回帰タスク用（MSELoss、R²評価）
- **SimpleNeuralNetwork**: 2層全結合ニューラルネットワーク（入力/隠れ/出力次元可変）

### バッチ推論（eval_batch.py）

- **evaluate_iris_batch**: 保存済みモデルとスケーラーを読み込み、複数サンプルを一括推論。分類はクラス名、回帰は数値を返却

### 結果保存（save_util.py）

- **store_model_and_learning_logs**: モデルパラメータ、学習曲線画像、CSV、実験管理ログを統合保存

## ファイル構成

```
src/machineLearning/
├── dataset.py                  # データセット定義と変換
├── pipeline.py                 # 訓練パイプライン
├── eval_batch.py               # バッチ推論
├── save_util.py                # 保存ユーティリティ
├── simple_nn.py                # 2層ニューラルネット
├── models/
│   ├── base_model.py           # 基底クラス
│   ├── classification_model.py # 分類モデル
│   └── regression_model.py     # 回帰モデル
└── examples/
    ├── train_iris.py           # Iris分類サンプル
    ├── train_diabetes.py       # Diabetes回帰サンプル
    └── ml_dataset_from_csv_sample.py
```

## 使用方法

### 基本的な訓練フロー

```python
from machineLearning.pipeline import train_and_save_pipeline
from machineLearning.dataset import MLDatasetConverter
from sklearn.datasets import load_iris

iris = load_iris()
iris_ds = MLDatasetConverter.convert(iris)

model, net, acc_hist, loss_hist, exp_name = train_and_save_pipeline(
    dataset=iris_ds, dataset_name="iris", epochs=50
)

test_score = model.evaluate_model()
print(f"Test accuracy: {test_score:.4f}")
```

### CSVからのデータセット読み込み

```python
from machineLearning.dataset import MLDatasetConverter
from pathlib import Path

dataset = MLDatasetConverter.from_csv(
    path=Path("data/machineLearning/custom_data.csv"),
    features=["feature1", "feature2", "feature3", "feature4"],
    target="label",
    dropna=True
)

model, net, acc_hist, loss_hist, exp_name = train_and_save_pipeline(
    dataset=dataset, dataset_name="custom_data", epochs=30, learning_rate=0.001
)
```

### バッチ推論

```python
from machineLearning.eval_batch import evaluate_iris_batch
from pathlib import Path

test_samples = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]

predictions = evaluate_iris_batch(
    input_data_list=test_samples,
    model_path=Path("outputs/machineLearning/20231122_120000/model_param.pth"),
    scaler_path=Path("outputs/machineLearning/20231122_120000/scaler.joblib"),
    class_names=["Iris setosa", "Iris versicolor", "Iris virginica"]
)
```

## 実行方法

### サンプルスクリプト

プロジェクトルートから実行：

```bash
uv run python -m src.machineLearning.examples.train_iris          # Iris分類
uv run python -m src.machineLearning.examples.train_diabetes      # Diabetes回帰
```

### Flask API経由

Flask server起動後、以下のエンドポイントを利用可能：

- `/api/iris-prediction`: 単一サンプル予測
- `/api/iris-batch-prediction`: バッチ予測

```bash
uv run run_app.py  # http://localhost:8000 で起動
```

## 出力ファイル構造

訓練実行後、以下の構造でファイルが保存されます：

```
outputs/machineLearning/
├── {実験名}/
│   ├── model_param.pth     # モデルパラメータ
│   ├── scaler.joblib       # StandardScaler
│   ├── loss_curve.png      # 損失曲線
│   ├── acc_curve.png       # 精度曲線
│   ├── loss.csv            # 損失ログ
│   └── acc.csv             # 精度ログ
└── train_log/
    └── trained_model.csv   # 全実験管理ログ
```

## コーディング規約

`.github/instructions/modern.instructions.md`に準拠：

- ファイル操作は`pathlib.Path`を使用（`os.path`不可）
- 全関数に型ヒント付与
- ロギングは`loguru.logger`を使用（標準`logging`不可）
- データバリデーションは`pydantic`を活用
- Python 3.13対応、型安全性重視

## トラブルシューティング

### モジュールインポートエラー

プロジェクトルートから`uv run run_app.py`を使用。`src/server/app.py`を直接実行すると、Pythonパスが正しく設定されずエラーが発生します。

### 訓練時のValueError（長さ不一致）

データセットの特徴量とターゲットの行数を確認。CSV読み込み時は`dropna=True`で欠損値行を削除可能。

### バッチ推論でファイル未検出

モデルパスを確認。保存先は`outputs/machineLearning/{実験名}/model_param.pth`。対応するスケーラーファイル（`scaler.joblib`）も必要。

### 分類精度が低い

- エポック数を増やす（推奨: 30以上）
- 学習率を調整（デフォルト: 0.01）
- クラスバランスを確認
- データセット分布を確認

### R²スコアが負

モデル予測性能が平均値を下回る状態。以下を試行：

- エポック数増加
- 学習率調整（0.001など）
- 隠れ層次元増加
- データ品質確認（外れ値、欠損値）

### GPU使用

現在はCPUベース。GPU対応には`models/base_model.py`と`simple_nn.py`で`torch.device`設定が必要。将来対応予定。

## よくある質問

### 独自ニューラルネットワークの使用

`simple_nn.py`を参考に`nn.Module`継承モデルを作成し、`models/base_model.py`の`neural_network_model`属性に設定。

### 損失関数・最適化手法の変更

`models/classification_model.py`または`models/regression_model.py`の`__init__`で`self.loss_criterion`と`self.optimizer`を変更。

### カスタムデータセット訓練

`MLDatasetConverter.from_csv()`またはpandas DataFrameから`MLDatasetConverter.from_dataframe()`を使用。形式は2次元特徴量配列（N×M）と1次元ターゲット配列（N）。

### 訓練済みモデル取得

訓練後、`outputs/machineLearning/{実験名}/`に保存。実験名は訓練時指定またはタイムスタンプ自動生成。

### モデルバージョン管理

`outputs/machineLearning/train_log/trained_model.csv`に全実験履歴を記録。実験名に意味のある名前を付けることで識別可能。

## 関連ドキュメント

- プロジェクト全体: `/.github/copilot-instructions.md`
- Flask API: `/src/server/README.md`
- データ配置: `/docs/dev_contract/データ配置.md`
- PyTorch: `/docs/tech_knowledge/torch.md`
- Pytest: `/docs/tech_knowledge/pytest.md` 
