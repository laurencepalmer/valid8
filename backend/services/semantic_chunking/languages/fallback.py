"""Fallback chunker for unsupported languages."""

from typing import Any, Optional

from backend.services.semantic_chunking.languages.base import LanguageConfig


class FallbackConfig(LanguageConfig):
    """Fallback configuration for languages without tree-sitter support."""

    @property
    def language_name(self) -> str:
        return "unknown"

    @property
    def file_extensions(self) -> list[str]:
        return []

    @property
    def tree_sitter_language(self) -> Any:
        return None

    @property
    def is_tree_sitter_supported(self) -> bool:
        return False

    @property
    def function_query(self) -> str:
        return ""

    @property
    def class_query(self) -> str:
        return ""

    @property
    def method_query(self) -> str:
        return ""

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        return {}

    def get_framework_patterns(self) -> dict[str, list[str]]:
        return {}
