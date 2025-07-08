"""Tests for convert_json.py module.

This module contains unit tests for the convert_json_to_two_dimensional_array function.
"""

from typing import Any, Dict, List

import pytest

from util.convert_json import convert_json_to_two_dimensional_array


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

        result = convert_json_to_two_dimensional_array(sample_data)
        expected = [["5.1", "3.5", "1.4", "0.2"]]

        assert result == expected

    def test_convert_with_custom_feature_keys(self) -> None:
        """Test conversion with custom feature keys."""
        sample_data: List[Dict[str, str]] = [
            {"name": "Alice", "age": "30", "city": "New York"}
        ]
        feature_keys = ["name", "age", "city"]

        result = convert_json_to_two_dimensional_array(sample_data, feature_keys)
        expected = [["Alice", "30", "New York"]]

        assert result == expected

    def test_convert_multiple_items(self) -> None:
        """Test conversion with multiple JSON objects."""
        sample_data: List[Dict[str, str]] = [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]
        feature_keys = ["a", "b"]

        result = convert_json_to_two_dimensional_array(sample_data, feature_keys)
        expected = [["1", "2"], ["3", "4"]]

        assert result == expected

    def test_empty_list_raises_error(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="json_data_list cannot be empty"):
            convert_json_to_two_dimensional_array([])

    def test_missing_key_raises_error(self) -> None:
        """Test that missing key in data raises KeyError."""
        sample_data: List[Dict[str, str]] = [{"a": "1"}]
        feature_keys = ["a", "missing_key"]

        with pytest.raises(KeyError):
            convert_json_to_two_dimensional_array(sample_data, feature_keys)

    def test_mixed_data_types(self) -> None:
        """Test conversion with mixed data types."""
        sample_data: List[Dict[str, Any]] = [
            {"str_val": "text", "int_val": 42, "float_val": 3.14}
        ]
        feature_keys = ["str_val", "int_val", "float_val"]

        result = convert_json_to_two_dimensional_array(sample_data, feature_keys)
        expected = [["text", 42, 3.14]]

        assert result == expected
