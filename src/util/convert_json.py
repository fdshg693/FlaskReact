from typing import Any, Dict, List, Optional

from loguru import logger


def convert_json_to_model_input(
    json_data_list: List[Dict[str, Any]], feature_keys: Optional[List[str]] = None
) -> List[List[float]]:
    """Convert a list of JSON objects to a 2D array of floats with specified column order.

    Args:
        json_data_list: List of dictionaries containing data
        feature_keys: List of key names to specify column order.
                     If None, defaults to iris feature keys.

    Returns:
        List of lists representing the data in 2D array format (all values as floats)

    Raises:
        KeyError: If a feature key is not found in the data
        ValueError: If json_data_list is empty or if values cannot be converted to float
    """
    if not json_data_list:
        raise ValueError("json_data_list cannot be empty")

    # Use provided feature keys or default to iris feature keys
    if feature_keys is None:
        feature_keys = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

    logger.info(f"Converting {len(json_data_list)} JSON objects to 2D array")
    logger.debug(f"Using feature keys: {feature_keys}")

    # Convert each dictionary to a list of float values following the specified key order
    two_dimensional_array: List[List[float]] = []
    for json_item in json_data_list:
        try:
            row: List[float] = [
                float(json_item[feature_key]) for feature_key in feature_keys
            ]
            two_dimensional_array.append(row)
        except KeyError as e:
            logger.error(f"Feature key {e} not found in data item: {json_item}")
            raise
        except (ValueError, TypeError) as e:
            logger.error(
                f"Cannot convert value to float in data item: {json_item}. Error: {e}"
            )
            raise ValueError(f"Cannot convert value to float: {e}") from e

    logger.info(
        f"Successfully converted to 2D array with {len(two_dimensional_array)} rows"
    )
    return two_dimensional_array


def main() -> None:
    """Main function to demonstrate the convert_json_to_model_input function."""
    # Sample iris flower data for testing
    sample_iris_data: List[Dict[str, str]] = [
        {
            "sepal.length": "5.1",
            "sepal.width": "3.5",
            "petal.length": "1.4",
            "petal.width": ".2",
        },
        {
            "petal.width": ".2",
            "sepal.width": "3",
            "petal.length": "1.4",
            "sepal.length": "4.9",
        },
        {
            "sepal.length": "4.7",
            "sepal.width": "3.2",
            "petal.length": "1.3",
            "petal.width": ".2",
        },
    ]

    feature_keys: List[str] = [
        "sepal.length",
        "sepal.width",
        "petal.length",
        "petal.width",
    ]

    try:
        converted_array = convert_json_to_model_input(
            sample_iris_data, feature_keys=feature_keys
        )
        logger.info(f"Converted array (floats only): {converted_array}")
    except (ValueError, KeyError) as e:
        logger.error(f"Error converting data: {e}")
        raise


if __name__ == "__main__":
    main()
