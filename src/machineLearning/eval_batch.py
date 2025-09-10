from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import StandardScaler

from machineLearning.simple_nn import SimpleNeuralNetwork


def evaluate_iris_batch(
    input_data_list: Sequence[Sequence[float]],
    model_path: Path | str,
    scaler_path: Path | str,
    class_names: Sequence[str] | None = None,
) -> List[str]:
    """汎用バッチ推論関数 (分類/回帰両対応)。

    互換性維持のため関数名は evaluate_iris_batch のまま。model の state_dict から
    入出力次元を自動推定し、出力次元=1 の場合は回帰とみなし数値予測を返す。

    分類: softmax -> argmax でクラス名 (または class_{i}) を返却
    回帰: モデル出力値(float) を文字列化して返却（呼び出し側が表示用途で文字列想定のため）

    Args:
        input_data_list: 2D シーケンス (samples x features)
        model_path: state_dict (.pth)
        scaler_path: StandardScaler (.joblib)
        class_names: 分類時のクラス名。None なら自動生成

    Returns:
        List[str]: 予測結果（分類: クラス名 / 回帰: 数値文字列）
    """
    # --- 早期検証 ---
    if not input_data_list:
        logger.error("入力データが空です。")
        return []

    # Path へ正規化
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    # --- スケーラー読込 & 入力次元取得 ---
    try:
        scaler: StandardScaler = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.error(f"スケーラーファイルが見つかりません: {scaler_path}")
        return []
    except Exception as e:  # pragma: no cover - 予期しない例外
        logger.error(f"スケーラー読込に失敗: {e}")
        return []

    # n_features_in_ (sklearn >=0.24) か mean_ の shape で判定
    try:
        expected_features = int(
            getattr(scaler, "n_features_in_", None) or scaler.mean_.shape[0]
        )
    except Exception as e:  # pragma: no cover
        logger.error(f"スケーラーから特徴量数を取得できません: {e}")
        return []

    # --- 各サンプル検証 (数値型 & 次元) ---
    for idx, sample in enumerate(input_data_list):
        if len(sample) != expected_features:
            logger.error(
                f"サンプル {idx} の特徴量数 {len(sample)} が想定 {expected_features} と一致しません"
            )
            return []
        if not all(isinstance(v, (int, float)) for v in sample):
            logger.error(f"サンプル {idx} に数値以外の値が含まれています: {sample}")
            return []

    # --- モデル state_dict の shape からネットワーク構築 ---
    if not model_path.exists():
        logger.error(f"モデルファイルが見つかりません: {model_path}")
        return []

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception as e:  # pragma: no cover
        logger.error(f"モデル state_dict の読み込みに失敗: {e}")
        return []

    try:
        # fully_connected_layer_1.weight: (hidden_dim, input_dim)
        fc1_w = state_dict["fully_connected_layer_1.weight"]
        fc2_w = state_dict["fully_connected_layer_2.weight"]
        inferred_input_dim = fc1_w.shape[1]
        hidden_dim = fc1_w.shape[0]
        output_dim = fc2_w.shape[0]
    except KeyError as e:
        logger.error(f"想定キーが state_dict に存在しません: {e}")
        return []

    if inferred_input_dim != expected_features:
        logger.error(
            f"入力特徴量数の不一致: scaler={expected_features}, model={inferred_input_dim}"
        )
        return []

    model = SimpleNeuralNetwork(
        input_dim=inferred_input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    try:
        model.load_state_dict(state_dict)
    except Exception as e:  # pragma: no cover
        logger.error(f"モデル state_dict の適用に失敗: {e}")
        return []
    model.eval()

    # --- 前処理 & Tensor 化 ---
    input_array = np.asarray(input_data_list, dtype=np.float32)
    try:
        scaled_array = scaler.transform(input_array)
    except Exception as e:
        logger.error(f"スケーリングに失敗: {e}")
        return []

    input_tensor = torch.tensor(scaled_array, dtype=torch.float32)
    # --- 推論 ---
    if output_dim == 1:
        # 回帰
        with torch.no_grad():
            preds = model(input_tensor).squeeze(1).tolist()
        str_preds = [f"{p:.6f}" for p in preds]
        logger.debug(
            f"Batch regression prediction completed | samples={len(str_preds)} features={expected_features}"
        )
        if class_names is not None:
            logger.warning("class_names は回帰モードでは無視されます")
        return str_preds
    # 分類
    if class_names is None:
        if output_dim == 3 and expected_features == 4:
            class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]
        else:
            class_names = [f"class_{i}" for i in range(output_dim)]
    else:
        if len(class_names) != output_dim:
            logger.error(
                f"class_names の長さ {len(class_names)} が モデル出力次元 {output_dim} と不一致"
            )
            return []
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).tolist()
    predictions = [str(class_names[i]) for i in pred_idx]
    logger.debug(
        f"Batch classification prediction completed | samples={len(predictions)} features={expected_features} output_dim={output_dim}"
    )
    return predictions


if __name__ == "__main__":
    try:
        from loguru import logger

        # デモ入力 (Iris 想定)
        input_data_list = [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.1, 4.7, 1.5],
        ]
        logger.info("--- デバッグ実行開始 ---")

        current_dir = Path(__file__).resolve().parent.parent  # project root 基準
        model_path = current_dir / "param" / "models_20250813_213851.pth"
        scaler_path = current_dir / "scaler" / "scaler.joblib"

        species = evaluate_iris_batch(input_data_list, model_path, scaler_path)
        logger.info(f"予測結果: {species}")
        logger.info("--- デバッグ実行終了 ---")
    except ValueError as e:
        logger.error(f"値エラー: {e}")
    except FileNotFoundError as e:
        logger.error(f"ファイルが見つかりません: {e}")
    except Exception as e:
        logger.exception(f"予期しないエラー: {e}")
    finally:
        logger.info("--- プログラム終了 ---")
