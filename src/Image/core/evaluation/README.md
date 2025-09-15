# 画像分類モデル評価システム

## 概要

このモジュールは、訓練済みの WoodNet 画像分類モデルの評価・推論機能を提供します。単一画像の予測、ディレクトリ一括評価、結果の可視化が可能です。

## 主要コンポーネント

### `evaluator.py` - ModelEvaluator クラス

訓練済みモデルの推論エンジンとして機能します。

```python
from image.core.evaluation.evaluator import ModelEvaluator

# 初期化
evaluator = ModelEvaluator(
    checkpoint_path="./checkpoints/experiment/best_accuracy.pth",
    device="cuda",  # または "cpu"
    img_size=128
)
```

#### 主要メソッド

**単一画像の予測**

```python
result = evaluator.predict_single_image("path/to/image.jpg")
# 返り値: {"predicted_class": 0, "confidence": 0.89, "probabilities": [0.89, 0.08, 0.03]}
```

**複数画像の一括予測**

```python
results = evaluator.predict_batch_images(["image1.jpg", "image2.jpg"])
# 各画像の予測結果リストを返す
```

**ディレクトリ一括評価**

```python
summary = evaluator.evaluate_directory("./dataset/test_images/", output_file="results.json")
# 統計情報とともに評価サマリーを返す
```

**予測結果の可視化**

```python
vis_img = evaluator.visualize_prediction("image.jpg", save_path="result.png", show_image=True)
# 画像と予測確率を可視化した画像を生成・保存
```

### `evaluate_model.py` - CLI インターフェース

コマンドライン及びプログラマティック評価のエントリーポイントです。

#### 関数 API

**利用可能チェックポイントの検索**

```python
from image.core.evaluation.evaluate_model import find_available_checkpoints

checkpoints = find_available_checkpoints("./checkpoints/")
# 利用可能なモデルチェックポイント一覧を返す
```

**メイン評価関数**

```python
from image.core.evaluation.evaluate_model import main

# 単一画像評価
result = main(
    model="./checkpoints/experiment/best_accuracy.pth",
    image="./test_image.jpg",
    visualize=True
)

# ディレクトリ一括評価
summary = main(
    model="./checkpoints/experiment/best_accuracy.pth",
    directory="./test_dataset/",
    output="./results.json"
)
```

#### コマンドライン使用例

```bash
# チェックポイント一覧表示
uv run python -m src.image.core.evaluation.evaluate_model --list-checkpoints

# 単一画像評価（可視化付き）
uv run python -m src.image.core.evaluation.evaluate_model \
    --model "./checkpoints/experiment/best_accuracy.pth" \
    --image "./test_image.jpg" \
    --visualize

# ディレクトリ一括評価
uv run python -m src.image.core.evaluation.evaluate_model \
    --model "./checkpoints/experiment/best_accuracy.pth" \
    --directory "./test_dataset/" \
    --output "./evaluation_results.json"
```

## 使用先・統合先

- **実験スクリプト**: `experiments/train_wood_classification.py` の訓練後評価
- **Flask API**: `server/app.py` のリアルタイム推論エンドポイント
- **Streamlit UI**: `streamlit/` アプリケーションのモデル評価機能
- **テストスイート**: モデル性能の自動検証パイプライン

このモジュールは、プロジェクト全体の推論・評価インフラとして機能し、訓練されたモデルの実用的な利用を可能にします。
