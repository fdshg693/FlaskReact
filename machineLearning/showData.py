from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from sklearn.datasets import load_iris, load_diabetes


def load_datasets() -> Tuple[Any, Any]:
    """Load iris and diabetes datasets from sklearn.

    Returns:
        Tuple containing iris and diabetes datasets.
    """
    iris_data = load_iris()
    diabetes_data = load_diabetes()
    return iris_data, diabetes_data


def save_diabetes_to_csv(diabetes_data: Any, output_path: Path) -> None:
    """Save diabetes dataset to CSV file.

    Args:
        diabetes_data: The diabetes dataset from sklearn.
        output_path: Path where the CSV file will be saved.
    """
    diabetes_df = pd.DataFrame(
        data=diabetes_data.data, columns=diabetes_data.feature_names
    )
    diabetes_df["target"] = diabetes_data.target

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diabetes_df.to_csv(output_path, index=False)
    print(f"Diabetes data saved to: {output_path}")


def main() -> None:
    """Main function to load datasets and save diabetes data to CSV."""
    iris_data, diabetes_data = load_datasets()

    # Define output path using pathlib
    current_dir = Path(__file__).parent
    save_path = current_dir / "../data/diabetes_data.csv"
    save_path = save_path.resolve()  # Convert to absolute path

    save_diabetes_to_csv(diabetes_data, save_path)


if __name__ == "__main__":
    main()
