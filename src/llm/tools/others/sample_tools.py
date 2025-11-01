from langchain_core.tools import tool
from typing import List


@tool
def add_numbers(nums: List[float]) -> float:
    """Add a list of numbers together."""
    return sum(nums)


if __name__ == "__main__":
    result = add_numbers.run({"nums": [1.5, 2.5, 3.0]})
    print(f"The sum is: {result}")  # The sum is: 7.0
