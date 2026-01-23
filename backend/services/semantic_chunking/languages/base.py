"""Abstract base class for language-specific configurations."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class LanguageConfig(ABC):
    """Abstract base class for language-specific semantic chunking configuration."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Name of the programming language."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """File extensions associated with this language."""
        pass

    @property
    @abstractmethod
    def tree_sitter_language(self) -> Any:
        """Return the tree-sitter language object."""
        pass

    @property
    def is_tree_sitter_supported(self) -> bool:
        """Whether this language has tree-sitter support."""
        return True

    @property
    @abstractmethod
    def function_query(self) -> str:
        """Tree-sitter query for finding function definitions."""
        pass

    @property
    @abstractmethod
    def class_query(self) -> str:
        """Tree-sitter query for finding class definitions."""
        pass

    @property
    @abstractmethod
    def method_query(self) -> str:
        """Tree-sitter query for finding method definitions."""
        pass

    @property
    def import_query(self) -> str:
        """Tree-sitter query for finding import statements."""
        return ""

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """
        Extract docstring from a function/class node.

        Args:
            node: The tree-sitter node
            source_bytes: The source code as bytes

        Returns:
            The docstring if found, None otherwise
        """
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """
        Extract the function/method signature.

        Args:
            node: The tree-sitter node
            source_bytes: The source code as bytes

        Returns:
            The signature string if applicable
        """
        return None

    def extract_decorators(self, node: Any, source_bytes: bytes) -> list[str]:
        """
        Extract decorators/annotations from a node.

        Args:
            node: The tree-sitter node
            source_bytes: The source code as bytes

        Returns:
            List of decorator strings
        """
        return []

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """
        Extract parameter names from a function/method node.

        Args:
            node: The tree-sitter node
            source_bytes: The source code as bytes

        Returns:
            List of parameter names
        """
        return []

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """
        Extract return type annotation if present.

        Args:
            node: The tree-sitter node
            source_bytes: The source code as bytes

        Returns:
            The return type as string, or None
        """
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """
        Return tree-sitter queries for detecting behavior patterns.

        Returns:
            Dict mapping behavior name to tree-sitter query
        """
        return {}

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """
        Return patterns for detecting framework usage.

        Returns:
            Dict mapping framework name to list of indicator patterns
        """
        return {}

    def get_node_text(self, node: Any, source_bytes: bytes) -> str:
        """Extract text from a tree-sitter node."""
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
