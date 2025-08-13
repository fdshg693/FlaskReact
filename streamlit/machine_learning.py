from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re
import sys

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from loguru import logger

# プロジェクトルートを設定（絶対インポートのため）
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class StreamlitMLApp:
    """Streamlit機械学習アプリケーション"""

    def __init__(self) -> None:
        """アプリケーションの初期化"""
        self.project_root = Path(__file__).resolve().parent.parent
        self.param_dir = self.project_root / "param"
        self.scaler_dir = self.project_root / "scaler"
        self.curve_log_dir = self.project_root / "curveLog"
        self.csv_log_dir = self.project_root / "csvLog"

    def validate_filename(self, filename: str) -> bool:
        """
        ファイル名の妥当性をチェック

        Args:
            filename: チェックするファイル名

        Returns:
            bool: 妥当性の結果
        """
        if not filename:
            return False
        # 禁止文字をチェック
        forbidden_chars = r'[<>:"/\\|?*]'
        if re.search(forbidden_chars, filename):
            return False
        # 予約語をチェック
        reserved_names = [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ]
        if filename.upper() in reserved_names:
            return False
        return True

    def get_existing_models(self) -> List[Dict[str, str]]:
        """
        train_log/trained_model.csvの内容を取得してモデル情報リストとして返す

        Returns:
            List[Dict[str, str]]: モデル情報のリスト
        """
        trained_model_csv = self.project_root / "train_log" / "trained_model.csv"
        models: List[Dict[str, str]] = []
        if trained_model_csv.exists():
            try:
                df = pd.read_csv(trained_model_csv)
                # 必要なカラムのみ抽出（name, accuracy, date, etc.）
                for _, row in df.iterrows():
                    model_info = {key: str(row[key]) for key in df.columns}
                    models.append(model_info)
            except Exception as e:
                logger.error(f"Error reading trained_model.csv: {e}")
        return models

    def get_existing_scalers(self) -> List[Dict[str, str]]:
        """
        既存のスケーラーの一覧を取得

        Returns:
            List[Dict[str, str]]: スケーラー情報のリスト
        """
        scalers = []

        if self.scaler_dir.exists():
            for scaler_file in self.scaler_dir.glob("*.joblib"):
                scaler_info = {
                    "name": scaler_file.stem,
                    "file": scaler_file.name,
                    "path": str(scaler_file),
                    "size": f"{scaler_file.stat().st_size / 1024:.2f} KB",
                    "modified": datetime.fromtimestamp(
                        scaler_file.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                }
                scalers.append(scaler_info)

        return sorted(scalers, key=lambda x: x["modified"], reverse=True)

    def execute_training_with_custom_name(
        self, dataset: object, file_suffix: Optional[str] = None
    ) -> Dict[str, any]:
        """
        カスタムファイル名で機械学習を実行

        Args:
            dataset: 学習に使用するデータセット
            file_suffix: カスタムファイル名（省略時はタイムスタンプ）

        Returns:
            Dict[str, any]: 実行結果
        """
        try:
            # 動的インポート（プロジェクトルートが設定された後）
            from machineLearning.ml_class import execute_machine_learning_pipeline

            # データセット名を判断
            dataset_name = "unknown"
            if hasattr(dataset, "filename") and "iris" in dataset.filename:
                dataset_name = "iris"
            elif hasattr(dataset, "filename") and "diabetes" in dataset.filename:
                dataset_name = "diabetes"
            elif hasattr(dataset, "DESCR") and "iris" in dataset.DESCR.lower():
                dataset_name = "iris"
            elif hasattr(dataset, "DESCR") and "diabetes" in dataset.DESCR.lower():
                dataset_name = "diabetes"

            # エポック数を設定（デフォルト値）
            epochs = 5

            # 機械学習パイプラインの実行
            classifier, model, accuracy_history, loss_history = (
                execute_machine_learning_pipeline(dataset, epochs, file_suffix)
            )

            # カスタム名の処理と学習結果の保存
            from machineLearning.save_util import (
                save_model_and_learning_curves_with_custom_name,
            )

            if file_suffix and self.validate_filename(file_suffix):
                file_suffix = save_model_and_learning_curves_with_custom_name(
                    model,
                    accuracy_history,
                    loss_history,
                    dataset_name,
                    epochs,
                    file_suffix,
                    self.project_root,
                )
            else:
                file_suffix = save_model_and_learning_curves_with_custom_name(
                    model,
                    accuracy_history,
                    loss_history,
                    dataset_name,
                    epochs,
                    None,
                    self.project_root,
                )

            # テスト精度の評価
            test_accuracy = classifier.evaluate_model()

            return {
                "success": True,
                "accuracy": test_accuracy,
                "timestamp": file_suffix,
                "message": f"学習が完了しました。テスト精度: {test_accuracy:.3f}",
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"学習中にエラーが発生しました: {e}",
            }

    def run_app(self) -> None:
        """Streamlitアプリケーションのメイン実行"""
        st.set_page_config(
            page_title="機械学習トレーニングアプリ", page_icon="🤖", layout="wide"
        )

        st.title("🤖 機械学習トレーニングアプリ")
        st.markdown("---")

        # サイドバーでデータセット選択
        st.sidebar.header("設定")
        dataset_choice = st.sidebar.selectbox(
            "データセットを選択:", ["Iris (アイリス)", "Diabetes (糖尿病)"]
        )

        # メインコンテンツを2列に分割
        col1, col2 = st.columns([1, 1])

        with col1:
            st.header("🚀 新しい学習の実行")

            # カスタムファイル名入力
            file_suffix = st.text_input(
                "ファイル名（任意）:",
                placeholder="例: my_model_v1",
                help="指定しない場合は自動的にタイムスタンプが使用されます",
            )

            # ファイル名の妥当性チェック
            if file_suffix:
                if self.validate_filename(file_suffix):
                    st.success("✅ 有効なファイル名です")
                else:
                    st.error(
                        "❌ 無効なファイル名です。特殊文字や予約語は使用できません。"
                    )

            # 学習実行ボタン
            if st.button("🎯 機械学習を実行", type="primary", use_container_width=True):
                # データセットの読み込み
                if dataset_choice == "Iris (アイリス)":
                    dataset = load_iris()
                    st.info("🌸 Irisデータセットで学習を開始します...")
                else:
                    dataset = load_diabetes()
                    st.info("🏥 Diabetesデータセットで学習を開始します...")

                # プログレスバー表示
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 学習実行
                with st.spinner("学習中..."):
                    status_text.text("学習を実行中...")
                    progress_bar.progress(50)

                    result = self.execute_training_with_custom_name(
                        dataset, file_suffix
                    )
                    progress_bar.progress(100)

                # 結果表示
                if result["success"]:
                    st.success(result["message"])
                    st.balloons()

                    # 詳細情報の表示
                    st.subheader("📊 学習結果")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("テスト精度", f"{result['accuracy']:.3f}")
                    with metric_col2:
                        st.metric("保存ID", result["timestamp"])

                    # 自動リフレッシュ（既存モデル一覧を更新）
                    st.rerun()
                else:
                    st.error(result["message"])

        with col2:
            st.header("📚 既存の学習済みモデル")

            # モデル一覧の取得と表示
            existing_models = self.get_existing_models()
            existing_scalers = self.get_existing_scalers()

            if existing_models:
                st.subheader("🏷️ 保存済みモデル (train_log/trained_model.csv)")
                model_df = pd.DataFrame(existing_models)
                # すべてのカラムを表示（CSVの内容に依存）
                st.dataframe(
                    model_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # 最新モデルの詳細表示（CSVの1行目）
                latest_model = existing_models[0]
                with st.expander(
                    f"📄 最新モデルの詳細: {latest_model.get('name', '不明')}"
                ):
                    for key, value in latest_model.items():
                        st.write(f"**{key}:** {value}")
            else:
                st.info(
                    "まだ学習済みモデルがありません。左側で新しい学習を実行してください。"
                )

            if existing_scalers:
                st.subheader("📐 保存済みスケーラー")
                scaler_df = pd.DataFrame(existing_scalers)
                st.dataframe(
                    scaler_df[["name", "size", "modified"]],
                    use_container_width=True,
                    hide_index=True,
                )

        # フッター情報
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                <p>🔧 powered by Streamlit | 🧠 機械学習パイプライン | 📁 自動ファイル保存</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """メイン関数"""
    app = StreamlitMLApp()
    app.run_app()


if __name__ == "__main__":
    main()
