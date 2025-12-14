from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from util.decorators import run_once


@run_once
def load_dotenv_workspace(env_path: Path | None = None, override=False) -> None:
    """
    .envファイルから環境変数を読み込む。
    run_onceデコレーターにより、リセットされない限り一度だけ実行される。

    Args:
        env_path: 空の場合は、プロジェクトルートの.envを使用する。
        override: Trueの場合、既存の環境変数を上書きする。デフォルトでは、上書きしない。

    Examples:
        load_dotenv_workspace() # 環境変数を読み込む（.envのオーバーライドはしない）
        load_dotenv_workspace() # デコレーターがキャッシュから結果を返そうとするため、.envは再度読み込まれない4
        load_dotenv_workspace(force=True) # デコレーターのキャッシュをクリアして、再度.envを読み込む

    Note:
        forceとoverrideは異なる概念です。forceはデコレーターのキャッシュを無視して関数を再実行するためのものであり、
        overrideはload_dotenv関数に渡され、既存の環境変数を上書きするかどうかを制御します。
    """
    if env_path is None:
        from config.paths import PATHS

        env_path = PATHS.project_root / ".env"

    load_dotenv(dotenv_path=env_path, override=override)
