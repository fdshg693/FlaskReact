"""Mock text splitter module for demonstration purposes."""

from typing import List

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Mock text splitting function."""
    return [
        f"テキストチャンク1: {text[:50]}... (ログイン機能デモ)",
        f"テキストチャンク2: {text[50:100]}... (認証が正常に動作しています)"
    ]