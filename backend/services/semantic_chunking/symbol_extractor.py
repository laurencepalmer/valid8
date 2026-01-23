"""Extract symbols (functions, classes, methods) from parsed ASTs."""

from dataclasses import dataclass
from typing import Optional

import tree_sitter

from backend.models.semantic_chunk import SymbolType, StructuralMetadata
from backend.services.semantic_chunking.ast_parser import get_ast_parser
from backend.services.semantic_chunking.languages import get_language_config
from backend.services.semantic_chunking.name_normalizer import normalize_symbol_name


@dataclass
class ExtractedSymbol:
    """Represents an extracted code symbol with its metadata."""

    node: tree_sitter.Node
    symbol_type: SymbolType
    name: str
    content: str
    metadata: StructuralMetadata
    docstring: Optional[str]
    parent_symbol: Optional["ExtractedSymbol"] = None


class SymbolExtractor:
    """Extract symbols from source code using tree-sitter AST."""

    def __init__(self):
        self.parser = get_ast_parser()

    def extract_symbols(
        self,
        source: str,
        language: str,
        file_path: str,
        relative_path: str,
    ) -> list[ExtractedSymbol]:
        """
        Extract all symbols from source code.

        Args:
            source: The source code string
            language: The programming language
            file_path: Absolute path to the file
            relative_path: Relative path for display

        Returns:
            List of extracted symbols
        """
        tree = self.parser.parse(source, language)
        if tree is None:
            return []

        source_bytes = source.encode("utf-8")
        config = get_language_config(language)
        symbols: list[ExtractedSymbol] = []

        # Extract classes first to establish parent relationships
        class_nodes = self.parser.find_classes(tree, language)
        class_map: dict[int, ExtractedSymbol] = {}

        for node in class_nodes:
            symbol = self._extract_class(
                node, source_bytes, config, language, file_path, relative_path
            )
            if symbol:
                symbols.append(symbol)
                class_map[node.id] = symbol

        # Extract standalone functions (not methods)
        function_nodes = self.parser.find_functions(tree, language)
        for node in function_nodes:
            # Skip if this function is inside a class
            if self._is_inside_class(node, class_nodes):
                continue
            symbol = self._extract_function(
                node, source_bytes, config, language, file_path, relative_path
            )
            if symbol:
                symbols.append(symbol)

        # Extract methods and associate with their parent class
        method_nodes = self.parser.find_methods(tree, language)
        for node in method_nodes:
            parent_class = self._find_parent_class(node, class_map)
            symbol = self._extract_method(
                node,
                source_bytes,
                config,
                language,
                file_path,
                relative_path,
                parent_class,
            )
            if symbol:
                symbols.append(symbol)

        return symbols

    def _extract_function(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        config,
        language: str,
        file_path: str,
        relative_path: str,
    ) -> Optional[ExtractedSymbol]:
        """Extract a function symbol."""
        name = self._get_symbol_name(node, source_bytes, config)
        if not name:
            return None

        start_line, end_line = self.parser.get_node_lines(node)
        content = self.parser.get_node_text(node, source_bytes)
        docstring = config.extract_docstring(node, source_bytes)
        signature = config.extract_signature(node, source_bytes)
        decorators = config.extract_decorators(node, source_bytes)
        parameters = config.extract_parameters(node, source_bytes)
        return_type = config.extract_return_type(node, source_bytes)

        metadata = StructuralMetadata(
            symbol_type=SymbolType.FUNCTION,
            symbol_name=name,
            qualified_name=name,
            file_path=file_path,
            relative_path=relative_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
        )

        return ExtractedSymbol(
            node=node,
            symbol_type=SymbolType.FUNCTION,
            name=name,
            content=content,
            metadata=metadata,
            docstring=docstring,
        )

    def _extract_class(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        config,
        language: str,
        file_path: str,
        relative_path: str,
    ) -> Optional[ExtractedSymbol]:
        """Extract a class symbol."""
        name = self._get_symbol_name(node, source_bytes, config)
        if not name:
            return None

        start_line, end_line = self.parser.get_node_lines(node)
        content = self.parser.get_node_text(node, source_bytes)
        docstring = config.extract_docstring(node, source_bytes)
        decorators = config.extract_decorators(node, source_bytes)

        metadata = StructuralMetadata(
            symbol_type=SymbolType.CLASS,
            symbol_name=name,
            qualified_name=name,
            file_path=file_path,
            relative_path=relative_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            decorators=decorators,
        )

        return ExtractedSymbol(
            node=node,
            symbol_type=SymbolType.CLASS,
            name=name,
            content=content,
            metadata=metadata,
            docstring=docstring,
        )

    def _extract_method(
        self,
        node: tree_sitter.Node,
        source_bytes: bytes,
        config,
        language: str,
        file_path: str,
        relative_path: str,
        parent_class: Optional[ExtractedSymbol],
    ) -> Optional[ExtractedSymbol]:
        """Extract a method symbol."""
        name = self._get_method_name(node, source_bytes, config)
        if not name:
            return None

        start_line, end_line = self.parser.get_node_lines(node)
        content = self.parser.get_node_text(node, source_bytes)
        docstring = config.extract_docstring(node, source_bytes)
        signature = config.extract_signature(node, source_bytes)
        decorators = config.extract_decorators(node, source_bytes)
        parameters = config.extract_parameters(node, source_bytes)
        return_type = config.extract_return_type(node, source_bytes)

        parent_scope = parent_class.name if parent_class else None
        qualified_name = f"{parent_scope}.{name}" if parent_scope else name

        metadata = StructuralMetadata(
            symbol_type=SymbolType.METHOD,
            symbol_name=name,
            qualified_name=qualified_name,
            file_path=file_path,
            relative_path=relative_path,
            language=language,
            start_line=start_line,
            end_line=end_line,
            parent_scope=parent_scope,
            signature=signature,
            decorators=decorators,
            parameters=parameters,
            return_type=return_type,
        )

        return ExtractedSymbol(
            node=node,
            symbol_type=SymbolType.METHOD,
            name=name,
            content=content,
            metadata=metadata,
            docstring=docstring,
            parent_symbol=parent_class,
        )

    def _get_symbol_name(
        self, node: tree_sitter.Node, source_bytes: bytes, config
    ) -> Optional[str]:
        """Get the name of a symbol from its node."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return config.get_node_text(child, source_bytes)
        return None

    def _get_method_name(
        self, node: tree_sitter.Node, source_bytes: bytes, config
    ) -> Optional[str]:
        """Get the name of a method from its node."""
        for child in node.children:
            if child.type in (
                "identifier",
                "property_identifier",
                "field_identifier",
            ):
                return config.get_node_text(child, source_bytes)
            # For some languages, the name is nested
            if child.type == "function_declarator":
                for sub in child.children:
                    if sub.type in ("identifier", "field_identifier"):
                        return config.get_node_text(sub, source_bytes)
        return None

    def _is_inside_class(
        self, node: tree_sitter.Node, class_nodes: list[tree_sitter.Node]
    ) -> bool:
        """Check if a node is inside any of the class nodes."""
        for class_node in class_nodes:
            if (
                node.start_byte >= class_node.start_byte
                and node.end_byte <= class_node.end_byte
            ):
                return True
        return False

    def _find_parent_class(
        self, node: tree_sitter.Node, class_map: dict[int, ExtractedSymbol]
    ) -> Optional[ExtractedSymbol]:
        """Find the parent class for a method node."""
        current = node.parent
        while current:
            for class_id, symbol in class_map.items():
                if (
                    node.start_byte >= symbol.node.start_byte
                    and node.end_byte <= symbol.node.end_byte
                ):
                    return symbol
            current = current.parent
        return None


# Global extractor instance
_symbol_extractor: Optional[SymbolExtractor] = None


def get_symbol_extractor() -> SymbolExtractor:
    """Get the global symbol extractor instance."""
    global _symbol_extractor
    if _symbol_extractor is None:
        _symbol_extractor = SymbolExtractor()
    return _symbol_extractor
