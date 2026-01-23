"""C language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_c as tsc

from backend.services.semantic_chunking.languages.base import LanguageConfig


class CConfig(LanguageConfig):
    """Configuration for C semantic chunking."""

    @property
    def language_name(self) -> str:
        return "c"

    @property
    def file_extensions(self) -> list[str]:
        return [".c", ".h"]

    @property
    def tree_sitter_language(self) -> Any:
        return tsc.language()

    @property
    def function_query(self) -> str:
        return """
        (function_definition
            declarator: (function_declarator
                declarator: (identifier) @name
                parameters: (parameter_list) @params
            )
            body: (compound_statement) @body
        ) @function
        """

    @property
    def class_query(self) -> str:
        # C doesn't have classes, but has struct definitions
        return """
        [
            (struct_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body
            ) @class

            (enum_specifier
                name: (type_identifier) @name
                body: (enumerator_list) @body
            ) @class

            (type_definition
                declarator: (type_identifier) @name
            ) @class
        ]
        """

    @property
    def method_query(self) -> str:
        # C doesn't have methods - return empty query
        return ""

    @property
    def import_query(self) -> str:
        return """
        (preproc_include) @import
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract C-style doc comment."""
        current = node.prev_sibling
        while current:
            if current.type == "comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("/**") or comment.startswith("/*"):
                    # Clean the comment
                    lines = comment.split("\n")
                    cleaned = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("/**") or line.startswith("/*"):
                            line = line[2:] if line.startswith("/*") else line[3:]
                        if line.endswith("*/"):
                            line = line[:-2]
                        if line.startswith("*"):
                            line = line[1:].strip()
                        if line:
                            cleaned.append(line)
                    return " ".join(cleaned).strip() or None
                elif comment.startswith("//"):
                    # Single line comment
                    return comment[2:].strip() or None
                break
            else:
                break
            current = current.prev_sibling
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function signature."""
        return_type = None
        name = None
        params = None

        for child in node.children:
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = self.get_node_text(child, source_bytes)
            elif child.type == "pointer_declarator":
                # Handle pointer return types
                for sub in child.children:
                    if sub.type == "function_declarator":
                        for fsub in sub.children:
                            if fsub.type == "identifier":
                                name = self.get_node_text(fsub, source_bytes)
                            elif fsub.type == "parameter_list":
                                params = self.get_node_text(fsub, source_bytes)
            elif child.type == "function_declarator":
                for sub in child.children:
                    if sub.type == "identifier":
                        name = self.get_node_text(sub, source_bytes)
                    elif sub.type == "parameter_list":
                        params = self.get_node_text(sub, source_bytes)

        if name and params:
            if return_type:
                return f"{return_type} {name}{params}"
            return f"{name}{params}"
        return None

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names."""
        params = []

        def find_params(n):
            for child in n.children:
                if child.type == "function_declarator":
                    for sub in child.children:
                        if sub.type == "parameter_list":
                            for param in sub.children:
                                if param.type == "parameter_declaration":
                                    for psub in param.children:
                                        if psub.type == "identifier":
                                            params.append(
                                                self.get_node_text(psub, source_bytes)
                                            )
                elif child.type == "pointer_declarator":
                    find_params(child)

        find_params(node)
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type."""
        for child in node.children:
            if child.type in (
                "primitive_type",
                "type_identifier",
                "sized_type_specifier",
            ):
                return self.get_node_text(child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for C behavior patterns."""
        return {
            "conditionals": "[( if_statement) (switch_statement) (conditional_expression)] @cond",
            "loops": "[( for_statement) (while_statement) (do_statement)] @loop",
            "goto": "(goto_statement) @goto",
            "malloc": "(call_expression function: (identifier) @fn (#match? @fn \"^(malloc|calloc|realloc|free)$\")) @memory",
            "pointer_ops": "(pointer_expression) @ptr",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting C framework usage."""
        return {
            "stdio": ["printf", "scanf", "fopen", "fclose", "fprintf"],
            "stdlib": ["malloc", "calloc", "realloc", "free", "exit"],
            "posix": ["pthread_", "fork", "exec", "pipe", "socket"],
            "glib": ["g_", "GObject", "GList", "GHashTable"],
        }
