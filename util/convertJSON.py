import json


def convert_json_to_two_dimensional_array(json_data_list, feature_keys=None):
    """Convert a list of JSON objects to a 2D array with specified column order.

    Args:
        json_data_list: List of dictionaries containing data
        feature_keys: List of key names to specify column order.
                     If None, defaults to iris feature keys.

    Returns:
        List of lists representing the data in 2D array format
    """
    # Use provided feature keys or default to iris feature keys
    if feature_keys is None:
        feature_keys = ["sepal.length", "sepal.width", "petal.length", "petal.width"]

    # Convert each dictionary to a list of values following the specified key order
    two_dimensional_array = list(
        map(
            lambda json_item: [json_item[feature_key] for feature_key in feature_keys],
            json_data_list,
        )
    )

    return two_dimensional_array


if __name__ == "__main__":
    # Sample iris flower data for testing
    sample_iris_data = [
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

    feature_keys = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    converted_array = convert_json_to_two_dimensional_array(
        sample_iris_data, feature_keys=feature_keys
    )
    print(converted_array)
