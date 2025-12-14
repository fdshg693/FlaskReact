"""
デコレーターを使った、柔軟なツール実装の実験例
このスクリプトの関数をそのまま利用することは非推奨
"""

from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from langchain_core.tools import BaseTool, tool
from loguru import logger

from config import PATHS

# Security constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory issues

T = TypeVar("T")


def decorator(func: Callable[..., T]) -> Callable[..., BaseTool]:
    """
    ツール関数をデコレートするデコレーター
    Args:
        func (Callable): ツール化したい関数。
            Args:
                search_directory_path (Optional[str]): 検索対象のディレクトリパス
                base_path (Path): ベースとなるドキュメントディレクトリパス（どの範囲のディレクトリを検索するかの制限）
    Returns:
        Callable: デコレーターでラップされたツール関数
    """

    def wrapper(
        base_path: Path = PATHS.llm_data / "text_documents",
    ) -> BaseTool:
        """
        base_pathを固定したツール関数を返す
        Args:
            base_path (Path): ベースとなるドキュメントディレクトリパス（どの範囲のディレクトリを検索するかの制限）
        Returns:
            Callable: ツール化されたエージェントから呼び出せる関数
        """

        # AIが直接認識する関数
        def search_local_text_tool(
            search_directory_path: Optional[Path | str] = None,
        ) -> str:
            """
            受け取ったディレクトリパスを元に、その配下にあるテキストファイルの情報を取得する
            Args:
                search_directory_path (Optional[Path | str]): 検索対象のディレクトリの絶対パスまたはbase_pathからの相対パス
            """
            # DocStringはAI用に専用のものを用意
            # デコレータでラップされた関数を呼び出す
            # Args:
            #     search_directory_path (Optional[Path | str]): 検索対象のディレクトリパス
            # Returns:
            #     T: 元の関数の戻り値の型
            return func(
                search_directory_path=search_directory_path, base_path=base_path
            )

        # toolを関数に適用した場合の戻り値の型はBaseToolになる
        return tool(search_local_text_tool)

    return wrapper


@decorator
def create_search_local_text_tool(
    search_directory_path: Optional[Path | str] = None,
    base_path: Path = PATHS.llm_data / "text_documents",
) -> str:
    """
    ローカルドキュメントディレクトリ内のテキストファイルを検索します。
    Args:
        search_directory_path (Optional[Path | str]): 検索対象のディレクトリパス。絶対パスまたはbase_pathからの相対パス。
        base_path (Path): ベースとなるドキュメントディレクトリパス（どの範囲のディレクトリを検索するかの制限）
    Returns:
        str: 検索結果のテキストリストの文字列形式
    """

    # デフォルトでは、base_path配下のtext_documentsディレクトリ配下のみを検索可能とする
    if not search_directory_path:
        search_directory_path = PATHS.llm_data / "text_documents"

    if isinstance(search_directory_path, str):
        search_directory_path = base_path / search_directory_path

    # 検索対象のディレクトリが存在しているか、ディレクトリであるかを確認
    if not search_directory_path.exists() and search_directory_path.is_dir():
        logger.info("Invalid search directory path provided.")
        return f"Invalid search directory: {search_directory_path}"

    # パスを解決（絶対パス化）
    try:
        search_directory_path = search_directory_path.resolve()
        base_path = base_path.resolve()
    except (OSError, ValueError) as e:
        logger.error(f"Path resolution error: {e}")
        return "Error: Invalid path format"

    # パストラバーサル攻撃対策
    try:
        if not str(search_directory_path).startswith(str(base_path)):
            logger.warning(f"Path traversal attempt detected: {search_directory_path}")
            return "Error: Invalid path detected for security reasons"
    except (OSError, ValueError) as e:
        logger.error(f"Path resolution error: {e}")
        return "Error: Invalid path format"

    # ディレクトリ配下にあるtxtファイルを再帰的に検索
    # 読み取りエラー・テキストファイルが存在しない場合は、早期リターン
    try:
        text_files_paths: list[Path] = sorted(search_directory_path.rglob("*.txt"))
        if not text_files_paths:
            return f"No text files found in the directory: {search_directory_path}"
    except (OSError, PermissionError) as error:
        return f"Error accessing directory {search_directory_path}: {error}"

    # Get relative paths from the base directory
    file_infos: List[str] = []
    for text_file_path in text_files_paths:
        try:
            relative_path: Path = text_file_path.relative_to(search_directory_path)
            file_size: int = text_file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File too large to read: {text_file_path}")
                continue
            size_str: str = (
                f"{file_size / 1024:.1f}KB"
                if file_size < 1024 * 1024
                else f"{file_size / (1024 * 1024):.1f}MB"
            )
            # パス情報とサイズ情報をリストに追加
            file_infos.append(f"{relative_path} ({size_str})")
        except (OSError, ValueError) as error:
            logger.error(f"Error processing file {text_file_path}: {error}")
            continue

    # ファイル情報のリストまたは、存在しない場合のメッセージを文字列で返す
    return "\n".join(file_infos) if file_infos else "No readable text files found."


if __name__ == "__main__":
    # 引数なしでテスト
    print("Testing with default path:")
    search_tool: BaseTool = create_search_local_text_tool()
    result = search_tool.run({})
    print(result)
    print("===" * 20)

    # 特定のサブディレクトリを指定してテスト
    print("\nTesting with specific subdirectory:")
    search_tool: BaseTool = create_search_local_text_tool(
        base_path=PATHS.llm_data / "text_documents"
    )
    # base_pathより上の階層にアクセスしようとしているので、警告が出ることを期待
    result = search_tool.run({"search_directory_path": PATHS.llm_data})
    print(result)
