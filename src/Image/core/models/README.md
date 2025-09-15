# Models モジュール

## 概要

画像分類タスクのためのニューラルネットワークモデルを提供します。抽象基底クラス `BaseModel` とその実装である `WoodNet` によって構成され、木材画像の分類に特化した CNN アーキテクチャを実装しています。

## クラス構成

### BaseModel

抽象基底クラスとして、全てのモデルの共通インターフェースを定義。

```python
from image.core.models.base_model import BaseModel

# 継承して新しいモデルを実装
class CustomModel(BaseModel):
    def __init__(self, num_class: int, **kwargs):
        super().__init__(num_class, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 順伝播処理を実装
        return output
```

**主要メソッド:**

- `get_model_info()`: モデル情報（パラメータ数、クラス数）を取得

### WoodNet

木材分類用 CNN モデル。動的な層数設定と高度な正規化機能を提供。

```python
from image.core.models.wood_net import WoodNet

# モデル初期化
model = WoodNet(
    num_class=3,        # クラス数
    img_size=128,       # 入力画像サイズ
    layer=3,            # 畳み込み層数
    num_hidden=4096,    # 全結合層ユニット数
    l2softmax=True,     # L2正規化の有効化
    dropout_rate=0.2    # ドロップアウト率
)

# 推論実行
output = model(input_tensor)  # (batch_size, num_class)
```

**アーキテクチャ特徴:**

- 動的な畳み込み層構築（設定可能な層数）
- Mish 活性化関数による高精度化
- L2 正規化 Softmax によるロバスト性向上
- ドロップアウトによる過学習防止

## 使用箇所

- **訓練**: `experiments/train_wood_classification.py` でモデル訓練
- **評価**: `core/evaluation/evaluator.py` で推論・評価
- **統合**: トレーナーとデータローダーと連携して学習パイプライン構築
