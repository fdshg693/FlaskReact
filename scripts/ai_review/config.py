"""
AI Review Configuration Module

環境変数の読み込み、検証、パス管理を行う共通設定モジュール
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class ConfigurationError(Exception):
    """設定関連のエラー"""

    pass


class ReviewConfig:
    """AI Review の設定を管理するクラス"""

    def __init__(self, env_file: Optional[Path] = None):
        """
        設定を初期化する

        Args:
            env_file: .envファイルのパス（省略時は自動検出）

        Raises:
            ConfigurationError: 必須の環境変数が設定されていない場合
        """
        # .envファイルの読み込み
        if env_file:
            load_dotenv(env_file)
        else:
            # カレントディレクトリから上位に向かって.envを探す
            load_dotenv()

        # プロジェクトルートの検出
        self._project_root = self._detect_project_root()

        # 環境変数の読み込みと検証
        self._load_and_validate()

    def _detect_project_root(self) -> Path:
        """
        プロジェクトルートディレクトリを自動検出する

        Returns:
            プロジェクトルートのPath

        検出ロジック:
        1. 環境変数 PROJECT_ROOT が設定されている場合はそれを使用
        2. .gitディレクトリを探して親ディレクトリを特定
        3. それも見つからない場合は現在のディレクトリ
        """
        # 環境変数から取得
        if os.getenv("PROJECT_ROOT"):
            return Path(os.getenv("PROJECT_ROOT")).resolve()

        # .gitディレクトリを探す
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists():
                return parent

        # 見つからない場合は現在のディレクトリ
        return current

    def _load_and_validate(self):
        """環境変数を読み込んでバリデーションする"""
        # 必須: OpenAI API Key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY が設定されていません。\n"
                ".env ファイルに 'OPENAI_API_KEY=your-api-key' を追加してください。\n"
                "APIキーは https://platform.openai.com/api-keys から取得できます。"
            )

        # オプション: AIモデル
        self.ai_model = os.getenv("AI_MODEL", "gpt-4o")

        # オプション: トークン数
        try:
            self.max_tokens = int(os.getenv("MAX_TOKENS", "10000"))
        except ValueError as e:
            raise ConfigurationError(
                f"MAX_TOKENS は整数である必要があります: {os.getenv('MAX_TOKENS')}"
            ) from e

        # オプション: Temperature
        try:
            self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        except ValueError as e:
            raise ConfigurationError(
                f"TEMPERATURE は数値である必要があります: {os.getenv('TEMPERATURE')}"
            ) from e

        # # パスの設定（TODO：本番は戻す）
        # self.tmp_dir = self.project_root / "tmp"
        # スクリプトディレクトリ配下のtmpフォルダ（TODO：本番は消すorコメントアウト）
        script_dir = Path(__file__).parent
        self.tmp_dir = script_dir / "tmp"

        # tmpディレクトリの作成
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_root(self) -> Path:
        """プロジェクトルートディレクトリ"""
        return self._project_root

    def get_diff_path(self) -> Path:
        """diff出力ファイルのパスを取得"""
        return self.tmp_dir / "diff.patch"

    def get_review_output_path(self) -> Path:
        """レビュー出力ファイルのパスを取得"""
        return self.tmp_dir / "ai_review_output.md"

    def validate(self) -> bool:
        """
        設定が正しいかを確認する

        Returns:
            検証結果（True: 正常, False: エラーあり）
        """
        try:
            # API Keyの存在確認
            if not self.openai_api_key:
                return False

            # トークン数の範囲確認
            if self.max_tokens <= 0:
                return False

            # Temperatureの範囲確認（0.0 ~ 2.0）
            if not (0.0 <= self.temperature <= 2.0):
                return False

            # プロジェクトルートの存在確認
            if not self.project_root.exists():
                return False

            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """設定の文字列表現（デバッグ用）"""
        return (
            f"ReviewConfig(\n"
            f"  project_root={self.project_root},\n"
            f"  ai_model={self.ai_model},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  tmp_dir={self.tmp_dir},\n"
            f"  openai_api_key={'***' if self.openai_api_key else 'Not set'}\n"
            f")"
        )


def main():
    """設定の動作確認用メイン関数"""
    try:
        config = ReviewConfig()
        print("✓ 設定の読み込みに成功しました")
        print(config)

        # バリデーション
        if config.validate():
            print("\n✓ 設定は正常です")
        else:
            print("\n✗ 設定に問題があります")

        # パス情報の表示
        print(f"\nDiff出力先: {config.get_diff_path()}")
        print(f"レビュー出力先: {config.get_review_output_path()}")

    except ConfigurationError as e:
        print(f"✗ 設定エラー: {e}")
        return 1
    except Exception as e:
        print(f"✗ 予期しないエラー: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
