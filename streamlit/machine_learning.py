from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import re
import time
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
        # 予測用（Iris専用）
        self.iris_feature_names = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]

    # --------------------------------------------------
    # ファイル取得ユーティリティ
    # --------------------------------------------------
    def get_param_models(self) -> List[Dict[str, str]]:
        """param/ ディレクトリ内の .pth モデル一覧を取得

        Returns:
            List[Dict[str, str]]: モデル情報のリスト
        """
        return self._collect_files(self.param_dir, "*.pth", name_mode="with_ext")

    def get_scalers(self) -> List[Dict[str, str]]:
        """
        既存のスケーラーの一覧を取得

        Returns:
            List[Dict[str, str]]: スケーラー情報のリスト
        """

        return self._collect_files(self.scaler_dir, "*.joblib", name_mode="stem")

    # --------------------------------------------------
    # 内部ファイル共通収集ヘルパー
    # --------------------------------------------------
    def _collect_files(
        self, directory: Path, pattern: str, name_mode: str = "stem"
    ) -> List[Dict[str, str]]:
        """指定ディレクトリからファイル情報を収集（重複ロジック集約）

        Args:
            directory: 走査対象ディレクトリ
            pattern: glob パターン (例: '*.pth')
            name_mode: 'stem' = 拡張子除去, 'with_ext' = 拡張子付き

        Returns:
            List[Dict[str, str]]: ファイル情報辞書リスト
        """
        items: List[Dict[str, str]] = []
        if not directory.exists():
            return items
        for file in directory.glob(pattern):
            if not file.is_file():
                continue
            try:
                name_value = file.stem if name_mode == "stem" else file.name
                items.append(
                    {
                        "name": name_value,  # UI 用表示名
                        "file": file.name,  # 常にフルファイル名提供
                        "path": str(file),
                        "size": f"{file.stat().st_size / 1024:.2f} KB",
                        "modified": datetime.fromtimestamp(
                            file.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to stat file {file}: {e}")
        # 日付降順
        return sorted(items, key=lambda x: x["modified"], reverse=True)

    # --------------------------------------------------
    # 削除ユーティリティ
    # --------------------------------------------------
    def _safe_delete(self, file_path: Path) -> bool:
        """指定ファイルを安全に削除

        Args:
            file_path: 削除対象

        Returns:
            bool: 成功可否
        """
        try:
            # param/ か scaler/ 配下のみ許可
            if not (
                str(file_path).startswith(str(self.param_dir))
                or str(file_path).startswith(str(self.scaler_dir))
            ):
                logger.warning(f"Skip delete (outside allowed dir): {file_path}")
                return False
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            logger.warning(f"File not found for delete: {file_path}")
            return False
        except Exception as e:  # noqa: BLE001
            logger.error(f"Delete failed {file_path}: {e}")
            return False

    def delete_models(self, names: List[str]) -> Dict[str, int]:
        """モデル(.pth)を削除

        Args:
            names: 削除対象ファイル名リスト

        Returns:
            Dict[str, int]: 結果統計
        """
        success = 0
        deleted_suffixes: List[str] = []
        for n in names:
            p = self.param_dir / n
            if self._safe_delete(p):
                success += 1
                # ファイル名から suffix 抽出 (models_{suffix}.pth)
                match = re.match(r"models_(.+)\.pth$", n)
                if match:
                    deleted_suffixes.append(match.group(1))
        if deleted_suffixes:
            self._remove_csv_rows_by_suffixes(deleted_suffixes)
        return {"requested": len(names), "deleted": success}

    def delete_scalers(self, names: List[str]) -> Dict[str, int]:
        """スケーラー(.joblib)を削除

        Args:
            names: 削除対象ファイル名リスト

        Returns:
            Dict[str, int]: 結果統計
        """
        success = 0
        for n in names:
            p = self.scaler_dir / n
            if self._safe_delete(p):
                success += 1
        return {"requested": len(names), "deleted": success}

    def _remove_csv_rows_by_suffixes(self, suffixes: List[str]) -> None:
        """trained_model.csv から指定 suffix の行を削除

        Args:
            suffixes: 削除対象 file_suffix のリスト
        """
        trained_model_csv = self.project_root / "train_log" / "trained_model.csv"
        if not trained_model_csv.exists():
            return
        try:
            df = pd.read_csv(trained_model_csv)
            if "file_suffix" not in df.columns:
                logger.warning(
                    "trained_model.csv に file_suffix カラムが無いため削除同期をスキップ"
                )
                return
            before = len(df)
            df = df[~df["file_suffix"].isin(suffixes)]
            after = len(df)
            if after != before:
                df.to_csv(trained_model_csv, index=False)
                logger.info(
                    f"trained_model.csv を更新: {before - after} 行削除 (suffix: {suffixes})"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"trained_model.csv 更新失敗 (削除同期): {e}")

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

            # エポック数 / 学習率: サイドバー設定が session_state にあれば利用
            epochs = 5
            learning_rate_override = None
            if "epochs_sidebar" in st.session_state:
                try:
                    epochs = int(st.session_state["epochs_sidebar"])
                except Exception:
                    pass
            if "lr_sidebar" in st.session_state:
                try:
                    lr_val = float(st.session_state["lr_sidebar"])
                    if lr_val > 0:
                        learning_rate_override = lr_val
                except Exception:
                    pass

            # 一貫したファイルサフィックスを事前決定（scaler と model を揃える）
            if file_suffix and self.validate_filename(file_suffix):
                effective_suffix = file_suffix
            else:
                effective_suffix = time.strftime("%Y%m%d_%H%M%S")

            # 機械学習パイプラインの実行（scaler を effective_suffix で保存）
            model_wrapper, model, accuracy_history, loss_history = (
                execute_machine_learning_pipeline(
                    dataset,
                    epochs,
                    f"_{effective_suffix}",
                    learning_rate=learning_rate_override,
                )
            )

            # 学習曲線 + モデル保存（既に suffix あるのでそのまま渡す）
            from machineLearning.save_util import (
                save_model_and_learning_curves_with_custom_name,
            )

            save_model_and_learning_curves_with_custom_name(
                model,
                accuracy_history,
                loss_history,
                dataset_name,
                epochs,
                effective_suffix,
                self.project_root,
            )

            # 評価メトリクス (分類: Accuracy, 回帰: R2)
            test_metric = model_wrapper.evaluate_model()
            metric_label = (
                "Accuracy"
                if getattr(model_wrapper, "is_classification", lambda: False)()
                else "R2"
            )

            return {
                "success": True,
                "accuracy": test_metric,  # 後方互換キー名を維持
                "metric_label": metric_label,
                "timestamp": effective_suffix,
                "message": f"学習が完了しました。{metric_label}: {test_metric:.3f}",
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

        st.session_state["epochs_sidebar"] = int(
            st.sidebar.number_input(
                "エポック数",
                min_value=1,
                max_value=200,
                value=5,
                step=1,
                help="学習エポック数",
            )
        )
        st.session_state["lr_sidebar"] = float(
            st.sidebar.number_input(
                "学習率 (learning rate)",
                min_value=1e-5,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.5f",
                help="分類初期値0.1 / 回帰0.01 推奨。ここで上書き。",
            )
        )
        st.sidebar.caption("回帰(糖尿病)では学習率を小さめに。")

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
                        label = result.get("metric_label", "テスト精度")
                        st.metric(label, f"{result['accuracy']:.3f}")
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
            existing_scalers = self.get_scalers()

            if existing_models:
                st.subheader("🏷️ 保存済みモデル (train_log/trained_model.csv)")
                model_df = pd.DataFrame(existing_models)
                # すべてのカラムを表示（CSVの内容に依存）
                st.dataframe(
                    model_df,
                    use_container_width=True,
                    hide_index=True,
                )

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

            # --------------------------------------------------
            # 削除 UI
            # --------------------------------------------------
            st.subheader("🗑️ モデル / スケーラー削除")
            tab_m, tab_s = st.tabs(["モデル削除", "スケーラー削除"])
            with tab_m:
                param_models = self.get_param_models()
                if not param_models:
                    st.info("param/ に .pth ファイルがありません")
                else:
                    selectable = [m["name"] for m in param_models]
                    sel_models = st.multiselect(
                        "削除するモデル (.pth) 選択",
                        selectable,
                        key="delete_models_select",
                    )
                    confirm_m = st.checkbox(
                        "確認: 選択したモデルを削除する", key="confirm_delete_models"
                    )
                    if st.button(
                        "選択モデルを削除",
                        type="secondary",
                        disabled=not (sel_models and confirm_m),
                    ):
                        result = self.delete_models(sel_models)
                        st.success(
                            f"モデル削除完了: {result['deleted']} / {result['requested']} 件"
                        )
                        st.rerun()
            with tab_s:
                existing_scalers_list = self.get_scalers()
                if not existing_scalers_list:
                    st.info("scaler/ に .joblib ファイルがありません")
                else:
                    selectable_s = [s["file"] for s in existing_scalers_list]
                    sel_scalers = st.multiselect(
                        "削除するスケーラー (.joblib) 選択",
                        selectable_s,
                        key="delete_scalers_select",
                    )
                    confirm_s = st.checkbox(
                        "確認: 選択したスケーラーを削除する",
                        key="confirm_delete_scalers",
                    )
                    if st.button(
                        "選択スケーラーを削除",
                        type="secondary",
                        disabled=not (sel_scalers and confirm_s),
                    ):
                        result = self.delete_scalers(sel_scalers)
                        st.success(
                            f"スケーラー削除完了: {result['deleted']} / {result['requested']} 件"
                        )
                        st.rerun()

        # --- 推論 (Iris) セクション ---
        st.markdown("---")
        with st.expander("🔍 Irisモデルでバッチ推論を行う", expanded=False):
            st.caption(
                "学習済みIrisモデル(.pth)とスケーラー(.joblib)を指定して複数行を一括予測します。"
            )

            # モデル & スケーラー ファイル列挙
            model_files = []
            if self.param_dir.exists():
                model_files = sorted(
                    [p for p in self.param_dir.glob("*.pth") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            scaler_files = []
            if self.scaler_dir.exists():
                scaler_files = sorted(
                    [p for p in self.scaler_dir.glob("*.joblib") if p.is_file()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

            if not model_files:
                st.info("param/ に .pth ファイルがありません。学習後に利用できます。")
            if not scaler_files:
                st.info(
                    "scaler/ に .joblib ファイルがありません。学習後に利用できます。"
                )

            if model_files and scaler_files:
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    selected_model = st.selectbox(
                        "モデル (.pth)",
                        model_files,
                        format_func=lambda p: p.name,
                    )
                with col_m2:
                    selected_scaler = st.selectbox(
                        "スケーラー (.joblib)",
                        scaler_files,
                        format_func=lambda p: p.name,
                    )

                # データ入力方法
                st.markdown("#### 入力データ (特徴量4列)")
                default_rows = 1
                n_rows = int(
                    st.number_input(
                        "行数", min_value=1, max_value=50, value=default_rows, step=1
                    )
                )

                # セッションステート初期化 & 行数変更時の保持ロジック
                # NOTE: Streamlitのポリシー上、ウィジェットkeyに対する st.session_state 直接代入は
                # StreamlitValueAssignmentNotAllowedError を引き起こすため、
                # 表示用(key)と内部保持用(key)を分離する。
                widget_key = "iris_predict_df_editor"  # data_editor 用
                storage_key = "iris_predict_df_data"  # 内部保持用 DataFrame

                if storage_key not in st.session_state:
                    # 初期化
                    st.session_state[storage_key] = pd.DataFrame(
                        [[0.0] * 4 for _ in range(n_rows)],
                        columns=self.iris_feature_names,
                    )
                else:
                    current_df = st.session_state[storage_key]
                    current_len = len(current_df)
                    # 行数増加: 既存値保持しつつゼロ行追加
                    if n_rows > current_len:
                        add_rows = pd.DataFrame(
                            [[0.0] * 4 for _ in range(n_rows - current_len)],
                            columns=self.iris_feature_names,
                        )
                        st.session_state[storage_key] = pd.concat(
                            [current_df, add_rows], ignore_index=True
                        )
                    # 行数減少: 先頭 n_rows 行のみ保持
                    elif n_rows < current_len:
                        st.session_state[storage_key] = current_df.iloc[
                            :n_rows
                        ].reset_index(drop=True)

                edited_df = st.data_editor(
                    st.session_state[storage_key],
                    use_container_width=True,
                    num_rows="dynamic",
                    key=widget_key,
                    hide_index=True,
                )
                # 編集結果を内部保持DataFrameに反映
                # st.session_state[storage_key] = edited_df.copy()

                predict_btn = st.button(
                    "🧪 予測を実行", type="primary", use_container_width=False
                )

                if predict_btn:
                    # 入力検証
                    if edited_df.isnull().any().any():
                        st.error(
                            "欠損値 (NaN) が含まれています。全て数値を入力してください。"
                        )
                    else:
                        # List[List[float]] に変換
                        try:
                            batch_data: List[List[float]] = (
                                edited_df[self.iris_feature_names]
                                .astype(float)
                                .values.tolist()
                            )
                        except Exception as e:  # 型変換エラー
                            st.error(f"数値変換エラー: {e}")
                            batch_data = []

                        if batch_data:
                            with st.spinner("推論中..."):
                                try:
                                    from machineLearning.eval_batch import (
                                        evaluate_iris_batch,
                                    )

                                    predictions = evaluate_iris_batch(
                                        batch_data,
                                        model_path=selected_model,
                                        scaler_path=selected_scaler,
                                    )
                                    if predictions:
                                        result_df = edited_df.copy()
                                        result_df["predicted_species"] = predictions
                                        st.success("推論が完了しました。")
                                        st.dataframe(
                                            result_df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    else:
                                        st.error(
                                            "推論に失敗しました。モデル/スケーラー/入力を確認してください。"
                                        )
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"推論中にエラーが発生しました: {e}")
                                    logger.error(f"Inference error: {e}")

        # --- Diabetes 回帰推論セクション ---
        st.markdown("---")
        with st.expander("🩺 Diabetesモデルでバッチ推論 (回帰)", expanded=False):
            st.caption(
                "学習済み回帰モデル(.pth)とスケーラー(.joblib)を選択し、10特徴量を入力して予測。"
            )

            # モデル & スケーラー ファイル列挙
            model_files_reg = (
                sorted(
                    [p for p in self.param_dir.glob("*.pth")],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if self.param_dir.exists()
                else []
            )
            scaler_files_reg = (
                sorted(
                    [p for p in self.scaler_dir.glob("*.joblib")],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if self.scaler_dir.exists()
                else []
            )

            if not model_files_reg:
                st.info("param/ に .pth がありません。")
            if not scaler_files_reg:
                st.info("scaler/ に .joblib がありません。")

            if model_files_reg and scaler_files_reg:
                rc1, rc2 = st.columns(2)
                with rc1:
                    selected_model_reg = st.selectbox(
                        "モデル (.pth) (回帰)",
                        model_files_reg,
                        format_func=lambda p: p.name,
                    )
                with rc2:
                    selected_scaler_reg = st.selectbox(
                        "スケーラー (.joblib) (回帰)",
                        scaler_files_reg,
                        format_func=lambda p: p.name,
                    )

                # データ入力方法
                st.markdown("#### 入力データ (特徴量10列)")
                default_rows_reg = 1
                n_rows_reg = int(
                    st.number_input(
                        "行数 (回帰)",
                        min_value=1,
                        max_value=50,
                        value=default_rows_reg,
                        step=1,
                    )
                )

                # セッションステート初期化 & 行数変更時の保持ロジック
                widget_key_reg = "diabetes_predict_df_editor"  # data_editor 用
                storage_key_reg = "diabetes_predict_df_data"  # 内部保持用 DataFrame

                if storage_key_reg not in st.session_state:
                    # 初期化
                    st.session_state[storage_key_reg] = pd.DataFrame(
                        [[0.0] * 10 for _ in range(n_rows_reg)],
                        columns=[f"f{i}" for i in range(10)],
                    )
                else:
                    cur_df = st.session_state[storage_key_reg]
                    cur_len = len(cur_df)
                    # 行数増加: 既存値保持しつつゼロ行追加
                    if n_rows_reg > cur_len:
                        add_rows = pd.DataFrame(
                            [[0.0] * 10 for _ in range(n_rows_reg - cur_len)],
                            columns=[f"f{i}" for i in range(10)],
                        )
                        st.session_state[storage_key_reg] = pd.concat(
                            [cur_df, add_rows], ignore_index=True
                        )
                    # 行数減少: 先頭 n_rows 行のみ保持
                    elif n_rows_reg < cur_len:
                        st.session_state[storage_key_reg] = cur_df.iloc[
                            :n_rows_reg
                        ].reset_index(drop=True)

                edited_df_reg = st.data_editor(
                    st.session_state[storage_key_reg],
                    use_container_width=True,
                    num_rows="dynamic",
                    key=widget_key_reg,
                    hide_index=True,
                )

                if st.button(
                    "🧪 回帰予測を実行", type="primary", use_container_width=False
                ):
                    # 入力検証
                    if edited_df_reg.isnull().any().any():
                        st.error(
                            "欠損値 (NaN) が含まれています。全て数値を入力してください。"
                        )
                    else:
                        # List[List[float]] に変換
                        try:
                            reg_batch = edited_df_reg.astype(float).values.tolist()
                        except Exception as e:  # 型変換エラー
                            st.error(f"数値変換エラー: {e}")
                            reg_batch = []

                        if reg_batch:
                            with st.spinner("推論中..."):
                                try:
                                    from machineLearning.eval_batch import (
                                        evaluate_iris_batch,
                                    )

                                    preds = evaluate_iris_batch(
                                        reg_batch,
                                        model_path=selected_model_reg,
                                        scaler_path=selected_scaler_reg,
                                    )
                                    if preds:
                                        out_df = edited_df_reg.copy()
                                        out_df["predicted_value"] = preds
                                        st.success("回帰推論が完了しました。")
                                        st.dataframe(
                                            out_df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                    else:
                                        st.error(
                                            "推論に失敗しました。モデル/スケーラー/入力を確認してください。"
                                        )
                                except Exception as e:
                                    st.error(f"推論中にエラーが発生しました: {e}")
                                    logger.error(f"Regression inference error: {e}")

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
