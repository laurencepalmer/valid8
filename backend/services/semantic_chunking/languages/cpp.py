"""C++ language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_cpp as tscpp

from backend.services.semantic_chunking.languages.base import LanguageConfig


class CppConfig(LanguageConfig):
    """Configuration for C++ semantic chunking."""

    @property
    def language_name(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"]

    @property
    def tree_sitter_language(self) -> Any:
        return tscpp.language()

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
        return """
        [
            (class_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body
            ) @class

            (struct_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body
            ) @class
        ]
        """

    @property
    def method_query(self) -> str:
        return """
        (field_declaration_list
            (function_definition
                declarator: (function_declarator
                    declarator: (field_identifier) @method_name
                    parameters: (parameter_list) @params
                )
                body: (compound_statement) @body
            ) @method
        )
        """

    @property
    def import_query(self) -> str:
        return """
        [
            (preproc_include) @import
            (using_declaration) @import
        ]
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract C++-style doc comment."""
        current = node.prev_sibling
        while current:
            if current.type == "comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("/**") or comment.startswith("///"):
                    lines = comment.split("\n")
                    cleaned = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("/**"):
                            line = line[3:]
                        elif line.startswith("///"):
                            line = line[3:]
                        if line.endswith("*/"):
                            line = line[:-2]
                        if line.startswith("*"):
                            line = line[1:].strip()
                        if line and not line.startswith("@") and not line.startswith("\\"):
                            cleaned.append(line)
                    return " ".join(cleaned).strip() or None
                break
            else:
                break
            current = current.prev_sibling
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function/method signature."""
        return_type = None
        name = None
        params = None
        qualifiers = []

        for child in node.children:
            if child.type in (
                "primitive_type",
                "type_identifier",
                "template_type",
                "qualified_identifier",
            ):
                return_type = self.get_node_text(child, source_bytes)
            elif child.type == "function_declarator":
                for sub in child.children:
                    if sub.type in ("identifier", "field_identifier"):
                        name = self.get_node_text(sub, source_bytes)
                    elif sub.type == "parameter_list":
                        params = self.get_node_text(sub, source_bytes)
            elif child.type == "virtual":
                qualifiers.append("virtual")
            elif child.type == "static":
                qualifiers.append("static")

        if name and params:
            parts = []
            if qualifiers:
                parts.extend(qualifiers)
            if return_type:
                parts.append(return_type)
            parts.append(f"{name}{params}")
            return " ".join(parts)
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

        find_params(node)
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type."""
        for child in node.children:
            if child.type in (
                "primitive_type",
                "type_identifier",
                "template_type",
                "qualified_identifier",
                "auto",
            ):
                return self.get_node_text(child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for C++ behavior patterns."""
        return {
            "conditionals": "[( if_statement) (switch_statement) (conditional_expression)] @cond",
            "loops": "[( for_statement) (while_statement) (do_statement) (for_range_loop)] @loop",
            "error_handling": "(try_statement) @try",
            "templates": "(template_declaration) @template",
            "lambda": "(lambda_expression) @lambda",
            "smart_pointers": "(call_expression function: (template_function) @fn (#match? @fn \"^(make_unique|make_shared)$\")) @smart_ptr",
            "raii": "(declaration declarator: (init_declarator)) @raii",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting C++ framework usage."""
        return {
            "stl": [
                "std::vector",
                "std::map",
                "std::string",
                "std::unique_ptr",
                "std::shared_ptr",
            ],
            "boost": ["boost::", "BOOST_"],
            "qt": ["Q_OBJECT", "QWidget", "QString", "QApplication", "signals:", "slots:"],
            "catch2": ["TEST_CASE", "REQUIRE", "CHECK", "SECTION"],
            "gtest": ["TEST", "TEST_F", "EXPECT_", "ASSERT_"],
        }
