"""Mock utility module for demonstration purposes."""

from typing import List, Dict, Any

def convert_json_to_two_dimensional_array(data: List[Dict[str, Any]]) -> List[List[float]]:
    """Mock JSON conversion function."""
    # Simple mock that returns some test data
    return [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3]
    ]