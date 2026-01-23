"""Tests for the name normalizer module."""

import pytest
from backend.services.semantic_chunking.name_normalizer import (
    normalize_symbol_name,
    tokens_to_readable,
)


class TestNormalizeSymbolName:
    """Tests for the normalize_symbol_name function."""

    def test_camel_case(self):
        """Test camelCase splitting."""
        assert normalize_symbol_name("getUserById") == ["get", "user", "by", "id"]
        assert normalize_symbol_name("parseJSON") == ["parse", "json"]
        assert normalize_symbol_name("myVariable") == ["my", "variable"]

    def test_pascal_case(self):
        """Test PascalCase splitting."""
        assert normalize_symbol_name("UserService") == ["user", "service"]
        assert normalize_symbol_name("HTTPResponse") == ["http", "response"]
        assert normalize_symbol_name("XMLParser") == ["xml", "parser"]

    def test_snake_case(self):
        """Test snake_case splitting."""
        assert normalize_symbol_name("get_user_by_id") == ["get", "user", "by", "id"]
        assert normalize_symbol_name("parse_json_data") == ["parse", "json", "data"]

    def test_screaming_snake_case(self):
        """Test SCREAMING_SNAKE_CASE splitting."""
        assert normalize_symbol_name("MAX_RETRY_COUNT") == ["max", "retry", "count"]
        assert normalize_symbol_name("HTTP_STATUS_CODE") == ["http", "status", "code"]

    def test_mixed_case(self):
        """Test mixed case patterns."""
        assert normalize_symbol_name("get_userById") == ["get", "user", "by", "id"]
        assert normalize_symbol_name("XMLHTTPRequest") == ["xmlhttp", "request"]

    def test_consecutive_uppercase(self):
        """Test handling of consecutive uppercase letters."""
        assert normalize_symbol_name("HTTPRequest") == ["http", "request"]
        assert normalize_symbol_name("parseXMLData") == ["parse", "xml", "data"]
        assert normalize_symbol_name("IOError") == ["io", "error"]

    def test_numbers(self):
        """Test handling of numbers in names."""
        assert normalize_symbol_name("getUser2") == ["get", "user", "2"]
        assert normalize_symbol_name("parse3DModel") == ["parse", "3", "d", "model"]

    def test_single_word(self):
        """Test single word names."""
        assert normalize_symbol_name("name") == ["name"]
        assert normalize_symbol_name("Name") == ["name"]
        assert normalize_symbol_name("NAME") == ["name"]

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_symbol_name("") == []

    def test_underscores_only(self):
        """Test names with leading/trailing underscores."""
        assert normalize_symbol_name("_private") == ["private"]
        assert normalize_symbol_name("__dunder__") == ["dunder"]

    def test_special_cases(self):
        """Test special edge cases."""
        assert normalize_symbol_name("ID") == ["id"]
        assert normalize_symbol_name("x") == ["x"]
        assert normalize_symbol_name("X") == ["x"]


class TestTokensToReadable:
    """Tests for the tokens_to_readable function."""

    def test_basic(self):
        """Test basic token joining."""
        assert tokens_to_readable(["get", "user", "by", "id"]) == "get user by id"

    def test_single_token(self):
        """Test single token."""
        assert tokens_to_readable(["name"]) == "name"

    def test_empty_list(self):
        """Test empty list."""
        assert tokens_to_readable([]) == ""
