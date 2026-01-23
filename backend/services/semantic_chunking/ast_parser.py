"""AST parser using tree-sitter for semantic code analysis."""

from typing import Any, Optional

import tree_sitter

from backend.services.semantic_chunking.languages import (
    LanguageConfig,
    get_language_config,
    is_supported_language,
)


class ASTParser:
    """Parser for creating and querying ASTs using tree-sitter."""

    def __init__(self):
        self._parsers: dict[str, tree_sitter.Parser] = {}
        self._languages: dict[str, tree_sitter.Language] = {}

    def _get_language(self, language: str) -> Optional[tree_sitter.Language]:
        """Get or create a tree-sitter Language for the given language."""
        if language not in self._languages:
            config = get_language_config(language)
            if not config.is_tree_sitter_supported:
                return None
            # Wrap the language capsule in tree_sitter.Language for compatibility
            lang_capsule = config.tree_sitter_language
            self._languages[language] = tree_sitter.Language(lang_capsule)
        return self._languages[language]

    def get_parser(self, language: str) -> Optional[tree_sitter.Parser]:
        """
        Get or create a tree-sitter parser for the given language.

        Args:
            language: The programming language name

        Returns:
            A tree-sitter Parser instance, or None if unsupported
        """
        if not is_supported_language(language):
            return None

        if language not in self._parsers:
            config = get_language_config(language)
            if not config.is_tree_sitter_supported:
                return None

            ts_language = self._get_language(language)
            if ts_language is None:
                return None
            parser = tree_sitter.Parser(ts_language)
            self._parsers[language] = parser

        return self._parsers[language]

    def parse(self, source: str, language: str) -> Optional[tree_sitter.Tree]:
        """
        Parse source code into an AST.

        Args:
            source: The source code string
            language: The programming language

        Returns:
            A tree-sitter Tree, or None if parsing fails
        """
        parser = self.get_parser(language)
        if parser is None:
            return None

        try:
            source_bytes = source.encode("utf-8")
            return parser.parse(source_bytes)
        except Exception:
            return None

    def query(
        self, tree: tree_sitter.Tree, query_string: str, language: str
    ) -> list[tuple[tree_sitter.Node, str]]:
        """
        Execute a tree-sitter query on the AST.

        Args:
            tree: The parsed AST
            query_string: The tree-sitter query string
            language: The programming language

        Returns:
            List of (node, capture_name) tuples
        """
        if not query_string or not query_string.strip():
            return []

        ts_language = self._get_language(language)
        if ts_language is None:
            return []

        try:
            query = tree_sitter.Query(ts_language, query_string)
            cursor = tree_sitter.QueryCursor(query)
            matches = list(cursor.matches(tree.root_node))
            # matches returns list of (pattern_idx, dict[capture_name, list[Node]])
            # Convert to list of (node, capture_name) tuples
            result = []
            for pattern_idx, captures in matches:
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        result.append((node, capture_name))
            return result
        except Exception:
            return []

    def find_functions(
        self, tree: tree_sitter.Tree, language: str
    ) -> list[tree_sitter.Node]:
        """Find all function definitions in the AST."""
        config = get_language_config(language)
        captures = self.query(tree, config.function_query, language)
        return [node for node, name in captures if name == "function"]

    def find_classes(
        self, tree: tree_sitter.Tree, language: str
    ) -> list[tree_sitter.Node]:
        """Find all class definitions in the AST."""
        config = get_language_config(language)
        captures = self.query(tree, config.class_query, language)
        return [node for node, name in captures if name == "class"]

    def find_methods(
        self, tree: tree_sitter.Tree, language: str
    ) -> list[tree_sitter.Node]:
        """Find all method definitions in the AST."""
        config = get_language_config(language)
        captures = self.query(tree, config.method_query, language)
        return [node for node, name in captures if name == "method"]

    def get_node_text(self, node: tree_sitter.Node, source_bytes: bytes) -> str:
        """Extract text from a tree-sitter node."""
        return source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def get_node_lines(self, node: tree_sitter.Node) -> tuple[int, int]:
        """Get the start and end line numbers of a node (1-indexed)."""
        return (node.start_point.row + 1, node.end_point.row + 1)


# Global parser instance
_ast_parser: Optional[ASTParser] = None


def get_ast_parser() -> ASTParser:
    """Get the global AST parser instance."""
    global _ast_parser
    if _ast_parser is None:
        _ast_parser = ASTParser()
    return _ast_parser
