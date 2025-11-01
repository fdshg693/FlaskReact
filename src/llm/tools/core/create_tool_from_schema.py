from langchain_core.tools import StructuredTool


def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(a + b)


tool = StructuredTool.from_function(add)
tool.run({"a": 1, "b": 2})  # 3
