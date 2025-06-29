from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from sklearn.datasets import load_iris, load_diabetes


def load_sklearn_datasets() -> Tuple[Any, Any]:
    """Load iris and diabetes datasets from sklearn.

    Returns:
        Tuple containing iris and diabetes datasets.
    """
    iris_dataset = load_iris()
    diabetes_dataset = load_diabetes()
    return iris_dataset, diabetes_dataset


def save_diabetes_dataset_to_csv(diabetes_dataset: Any, csv_output_path: Path) -> None:
    """Save diabetes dataset to CSV file.

    Args:
        diabetes_dataset: The diabetes dataset from sklearn.
        csv_output_path: Path where the CSV file will be saved.
    """
    diabetes_dataframe = pd.DataFrame(
        data=diabetes_dataset.data, columns=diabetes_dataset.feature_names
    )
    diabetes_dataframe["target"] = diabetes_dataset.target

    # Ensure parent directory exists
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    diabetes_dataframe.to_csv(csv_output_path, index=False)
    print(f"Diabetes data saved to: {csv_output_path}")


def main() -> None:
    """Main function to load datasets and save diabetes data to CSV."""
    iris_dataset, diabetes_dataset = load_sklearn_datasets()

    # Define output path using pathlib
    script_directory = Path(__file__).parent
    diabetes_csv_path = script_directory / "../data/diabetes_data.csv"
    diabetes_csv_path = diabetes_csv_path.resolve()  # Convert to absolute path

    save_diabetes_dataset_to_csv(diabetes_dataset, diabetes_csv_path)


if __name__ == "__main__":
    main()
