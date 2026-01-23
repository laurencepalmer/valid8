"""Normalize symbol names by splitting camelCase, PascalCase, snake_case, etc."""

import re


def normalize_symbol_name(name: str) -> list[str]:
    """
    Split a symbol name into normalized tokens.

    Examples:
        normalize_symbol_name("getUserById") -> ["get", "user", "by", "id"]
        normalize_symbol_name("HTTPRequest") -> ["http", "request"]
        normalize_symbol_name("parse_json_data") -> ["parse", "json", "data"]
        normalize_symbol_name("MAX_RETRY_COUNT") -> ["max", "retry", "count"]
    """
    if not name:
        return []

    # Handle SCREAMING_SNAKE_CASE first
    if name.isupper() and "_" in name:
        return [part.lower() for part in name.split("_") if part]

    # Handle snake_case
    if "_" in name:
        parts = []
        for part in name.split("_"):
            if part:
                # Recursively handle camelCase within snake_case parts
                parts.extend(_split_camel_case(part))
        return [p.lower() for p in parts if p]

    # Handle camelCase and PascalCase
    return [p.lower() for p in _split_camel_case(name) if p]


def _split_camel_case(name: str) -> list[str]:
    """
    Split camelCase and PascalCase names.

    Handles consecutive uppercase letters (like HTTP, API, ID).
    """
    if not name:
        return []

    # Pattern to split on:
    # - Before an uppercase letter that is followed by lowercase
    # - After a sequence of uppercase letters before a lowercase
    # - Between lowercase/digit and uppercase
    parts = re.split(
        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Za-z])|(?<=[A-Za-z])(?=[0-9])",
        name,
    )

    result = []
    for part in parts:
        if part:
            result.append(part)

    return result


def tokens_to_readable(tokens: list[str]) -> str:
    """
    Convert normalized tokens to a readable string.

    Example:
        tokens_to_readable(["get", "user", "by", "id"]) -> "get user by id"
    """
    return " ".join(tokens)
