from typing import List

from langchain_core.tools import tool


@tool
def add_numbers(nums: List[float]) -> float:
    """Add a list of numbers together."""
    print("=" * 20)
    print("add_numbers called")
    print(f"Adding numbers: {nums}")
    print("=" * 20)
    return sum(nums)


if __name__ == "__main__":
    result = add_numbers.run({"nums": [1.5, 2.5, 3.0]})
    print(f"The sum is: {result}")  # The sum is: 7.0
