"""Tests for convert_json.py module.

This module contains unit tests for the convert_json_to_two_dimensional_array function.
"""

from typing import Any, Dict, List

import pytest

from util.convert_json import convert_json_to_model_input


class TestConvertJsonToTwoDimensionalArray:
    """Test class for convert_json_to_two_dimensional_array function."""

    def test_convert_with_default_feature_keys(self) -> None:
        """Test conversion with default iris feature keys."""
        sample_data: List[Dict[str, str]] = [
            {
                "sepal.length": "5.1",
                "sepal.width": "3.5",
                "petal.length": "1.4",
                "petal.width": "0.2",
            }
        ]

        result = convert_json_to_model_input(sample_data)
        expected = [[5.1, 3.5, 1.4, 0.2]]

        assert result == expected

    def test_convert_with_custom_feature_keys(self) -> None:
        """Test conversion with custom numeric feature keys."""
        sample_data: List[Dict[str, str]] = [
            {"height": "180.5", "weight": "75.2", "age": "30"}
        ]
        feature_keys = ["height", "weight", "age"]

        result = convert_json_to_model_input(sample_data, feature_keys)
        expected = [[180.5, 75.2, 30.0]]

        assert result == expected

    def test_convert_multiple_items(self) -> None:
        """Test conversion with multiple JSON objects."""
        sample_data: List[Dict[str, str]] = [
            {"a": "1.5", "b": "2.7"},
            {"a": "3.1", "b": "4.8"},
        ]
        feature_keys = ["a", "b"]

        result = convert_json_to_model_input(sample_data, feature_keys)
        expected = [[1.5, 2.7], [3.1, 4.8]]

        assert result == expected

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="json_data_list cannot be empty"):
            convert_json_to_model_input([])

    def test_missing_key_raises_error(self) -> None:
        """Test that missing key in data raises KeyError."""
        sample_data: List[Dict[str, str]] = [{"a": "1"}]
        feature_keys = ["a", "missing_key"]

        with pytest.raises(KeyError):
            convert_json_to_model_input(sample_data, feature_keys)

    def test_mixed_data_types(self) -> None:
        """Test conversion with mixed numeric data types."""
        sample_data: List[Dict[str, Any]] = [
            {"int_val": 42, "float_val": 3.14, "str_val": "2.5"}
        ]
        feature_keys = ["int_val", "float_val", "str_val"]

        result = convert_json_to_model_input(sample_data, feature_keys)
        expected = [[42.0, 3.14, 2.5]]

        assert result == expected

    def test_non_numeric_value_raises_error(self) -> None:
        """Test that non-numeric values raise ValueError."""
        sample_data: List[Dict[str, str]] = [{"value": "not_a_number"}]
        feature_keys = ["value"]

        with pytest.raises(ValueError, match="Cannot convert value to float"):
            convert_json_to_model_input(sample_data, feature_keys)
