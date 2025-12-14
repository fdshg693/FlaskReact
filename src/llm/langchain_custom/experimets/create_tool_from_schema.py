"""
langchainの内部機能を利用した、低レイヤーのツール作成サンプル（まだ未運用）
"""

from langchain_core.tools import StructuredTool


def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(a + b)


if __name__ == "__main__":
    tool = StructuredTool.from_function(add)
    tool.run({"a": 1, "b": 2})  # 3
