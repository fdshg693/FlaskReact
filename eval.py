import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

import numpy as np # 新しいデータを扱うためにnumpyをインポート


# (SimpleNet クラスの定義は前回の回答と同じなので省略します)
class SimpleNet(nn.Module):
   def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
       super().__init__()
       self.fc1 = nn.Linear(input_dim, hidden_dim)
       self.fc2 = nn.Linear(hidden_dim, output_dim)
   def forward(self, x):
       x = F.relu(self.fc1(x))
       return self.fc2(x)


# --- モデルを読み込む ---
loaded_model = SimpleNet()
param_dir = "param"
model_path = os.path.join(param_dir, "models.pth")


if os.path.exists(model_path):
   loaded_model.load_state_dict(torch.load(model_path))
   print(f"モデルのパラメータを {model_path} から読み込みました。")
else:
   print(f"エラー: パス {model_path} にモデルファイルが見つかりません。処理を中断します。")
   exit() # モデルがない場合は終了


# --- 評価モードに設定 ---
loaded_model.eval()
print("モデルを評価モードに設定しました。")


# --- 新しいデータで予測を行う ---


# 1. スケーラーの準備
#    重要: 訓練時に使用したスケーラーのインスタンスが必要です。
#    元の MachineLearning クラスのインスタンスがまだ利用可能な場合:
#    scaler = machineLearning.scaler
#
#    もし、別のスクリプトやセッションで実行している場合、
#    訓練時にスケーラーを保存しておく必要があります。
#    例: import joblib
#         joblib.dump(machineLearning.scaler, 'scaler.gz')
#    そして、ここで読み込みます:
#    scaler = joblib.load('scaler.gz')
#
#    ここでは、仮に新しい StandardScaler インスタンスを作成しますが、
#    実際には訓練時のものを使用してください。
#    以下の `scaler` は、あくまでデモンストレーション用です。
#    実際のシナリオでは、訓練済みの `scaler` をロードするか、
#    `machineLearning.scaler` を参照してください。
#   
#    もし `machineLearning` オブジェクトがまだスコープ内にあるなら、以下のようにします。
#    (最初のコードブロックを実行したPythonセッション内であると仮定)
try:
   # machineLearning は最初のスクリプトで定義されたインスタンス名と仮定
   scaler = machineLearning.scaler
   print("訓練時のスケーラーを使用します。")
except NameError:
   print("警告: 訓練時の 'machineLearning.scaler' が見つかりません。")
   print("       予測のためには、訓練に使用したのと同じスケーラーで新しいデータを変換する必要があります。")
   print("       ダミーのスケーラーを作成しますが、これは正しい結果を保証しません。")
   # 実際のアプリケーションでは、ここでエラーにするか、
   # 保存されたスケーラーをロードする処理を実装してください。
   scaler = StandardScaler() # これは本来行うべきではありません。
                           # fitされていないスケーラーは意味がありません。
                           # しかし、コードを動かすための一時的な措置です。
                           # 実際には、訓練済みスケーラーをロードしてください。
                           # 例: from joblib import load
                           #     scaler = load('path/to/your/saved_scaler.pkl')


# 2. 新しいデータの準備 (例)
#    これはIrisデータセットの形式 (4つの特徴量) に合わせたダミーデータです。
#    実際のデータに置き換えてください。
new_data_numpy = np.array([
   [5.1, 3.5, 1.4, 0.2],  # 1つ目のサンプル
   [6.7, 3.1, 4.4, 1.4],  # 2つ目のサンプル
   [7.0, 3.2, 6.0, 2.0]   # 3つ目のサンプル
], dtype=np.float32)


print(f"\n新しいデータ (変換前):\n{new_data_numpy}")


# 3. 新しいデータの前処理 (スケーリング)
#    重要: 訓練データに対しては .fit_transform() でしたが、
#          新しいデータに対しては .transform() を使います。
#          これは、訓練データから学習した平均と標準偏差を使って変換するためです。
#    注意: 上記の `scaler` の準備が適切でない場合、この結果は意味を持ちません。
try:
   new_data_scaled_numpy = scaler.transform(new_data_numpy)
   print(f"新しいデータ (スケーリング後):\n{new_data_scaled_numpy}")
except AttributeError:
   print("エラー: スケーラーが正しく初期化されていないか、'transform' メソッドがありません。")
   print("       訓練時のスケーラーを正しくロードまたは参照しているか確認してください。")
   exit()
except Exception as e: # NotFittedErrorなど sklearn.exceptions.NotFittedError
   print(f"スケーリングエラー: {e}")
   print("       スケーラーが訓練データで 'fit' されていない可能性があります。")
   print("       訓練時のスケーラーを正しくロードまたは参照しているか確認してください。")
   exit()




# 4. PyTorch Tensor に変換
new_data_tensor = torch.tensor(new_data_scaled_numpy, dtype=torch.float32)
print(f"新しいデータ (Tensor):\n{new_data_tensor}")


# 5. モデルで予測を実行
with torch.no_grad(): # 勾配計算をオフにして、メモリ効率を上げ、計算を高速化
   predictions = loaded_model(new_data_tensor)


print(f"モデルの出力 (ロジット/確率):\n{predictions}")


# 6. 予測結果の解釈 (最も確率の高いクラスを取得)
#    CrossEntropyLoss を使ったモデルの出力は通常、各クラスのロジット (logit) です。
#    これを softmax に通すと確率になりますが、argmax で最大値のインデックスを取得するだけなら不要です。
predicted_classes_indices = torch.argmax(predictions, dim=1)
print(f"予測されたクラスのインデックス:\n{predicted_classes_indices}")


# (オプション) Irisデータセットのクラス名にマッピングする場合
# target_names = load_iris().target_names # 元のデータセットからクラス名を取得
# predicted_class_names = [target_names[i] for i in predicted_classes_indices]
# print(f"予測されたクラス名:\n{predicted_class_names}")
