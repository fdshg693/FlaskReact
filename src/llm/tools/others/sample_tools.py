from langchain_core.tools import tool
from typing import List


@tool
def add_numbers(nums: List[float]) -> float:
    """Add a list of numbers together."""
    return sum(nums)
