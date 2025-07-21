"""Mock machine learning module for demonstration purposes."""

from typing import List

def evaluate_iris_batch(data: List[List[float]]) -> List[str]:
    """Mock iris prediction function."""
    results = []
    for i, row in enumerate(data):
        species = ["setosa", "versicolor", "virginica"][i % 3]
        results.append(f"{species} (ログイン認証済み)")
    return results