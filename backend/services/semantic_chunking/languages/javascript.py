"""JavaScript language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_javascript as tsjavascript

from backend.services.semantic_chunking.languages.base import LanguageConfig


class JavaScriptConfig(LanguageConfig):
    """Configuration for JavaScript semantic chunking."""

    @property
    def language_name(self) -> str:
        return "javascript"

    @property
    def file_extensions(self) -> list[str]:
        return [".js", ".jsx", ".mjs", ".cjs"]

    @property
    def tree_sitter_language(self) -> Any:
        return tsjavascript.language()

    @property
    def function_query(self) -> str:
        return """
        [
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @function

            (variable_declarator
                name: (identifier) @name
                value: (arrow_function
                    parameters: (formal_parameters) @params
                    body: (_) @body
                )
            ) @function

            (variable_declarator
                name: (identifier) @name
                value: (function_expression
                    parameters: (formal_parameters) @params
                    body: (statement_block) @body
                )
            ) @function

            (export_statement
                (function_declaration
                    name: (identifier) @name
                    parameters: (formal_parameters) @params
                    body: (statement_block) @body
                ) @function
            )
        ]
        """

    @property
    def class_query(self) -> str:
        return """
        [
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body
            ) @class

            (export_statement
                (class_declaration
                    name: (identifier) @name
                    body: (class_body) @body
                ) @class
            )
        ]
        """

    @property
    def method_query(self) -> str:
        return """
        (class_body
            (method_definition
                name: (property_identifier) @method_name
                parameters: (formal_parameters) @params
                body: (statement_block) @body
            ) @method
        )
        """

    @property
    def import_query(self) -> str:
        return """
        [
            (import_statement) @import
            (export_statement) @export
        ]
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract JSDoc comment from JavaScript function/class."""
        # Look for preceding comment
        current = node.prev_sibling
        while current:
            if current.type == "comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("/**"):
                    # Clean JSDoc comment
                    lines = comment.split("\n")
                    cleaned = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("/**"):
                            line = line[3:]
                        if line.endswith("*/"):
                            line = line[:-2]
                        if line.startswith("*"):
                            line = line[1:].strip()
                        if line and not line.startswith("@"):
                            cleaned.append(line)
                    return " ".join(cleaned).strip() or None
                break
            elif current.type not in ("export_statement",):
                break
            current = current.prev_sibling
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function signature."""
        # For function declarations
        if node.type == "function_declaration":
            name = None
            params = None
            for child in node.children:
                if child.type == "identifier":
                    name = self.get_node_text(child, source_bytes)
                elif child.type == "formal_parameters":
                    params = self.get_node_text(child, source_bytes)
            if name and params:
                return f"function {name}{params}"

        # For variable declarators with arrow/function expressions
        elif node.type == "variable_declarator":
            name = None
            params = None
            for child in node.children:
                if child.type == "identifier":
                    name = self.get_node_text(child, source_bytes)
                elif child.type in ("arrow_function", "function_expression"):
                    for sub in child.children:
                        if sub.type == "formal_parameters":
                            params = self.get_node_text(sub, source_bytes)
                            break
            if name and params:
                return f"const {name} = {params} =>"

        return None

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names."""
        params = []

        def find_params(n):
            for child in n.children:
                if child.type == "formal_parameters":
                    for param in child.children:
                        if param.type == "identifier":
                            params.append(self.get_node_text(param, source_bytes))
                        elif param.type in (
                            "assignment_pattern",
                            "rest_pattern",
                            "object_pattern",
                        ):
                            for sub in param.children:
                                if sub.type == "identifier":
                                    params.append(self.get_node_text(sub, source_bytes))
                                    break
                elif child.type in ("arrow_function", "function_expression"):
                    find_params(child)

        find_params(node)
        return params

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for JavaScript behavior patterns."""
        return {
            "conditionals": "[( if_statement) (ternary_expression)] @cond",
            "loops": "[( for_statement) (for_in_statement) (while_statement) (do_statement)] @loop",
            "error_handling": "(try_statement) @try",
            "async": "[( async) (await_expression)] @async",
            "promises": "(call_expression function: (member_expression property: (property_identifier) @prop (#match? @prop \"^(then|catch|finally)$\"))) @promise",
            "callbacks": "(arrow_function) @arrow",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting JavaScript framework usage."""
        return {
            "react": [
                "useState",
                "useEffect",
                "useContext",
                "useRef",
                "useMemo",
                "useCallback",
                "React.Component",
                "jsx",
            ],
            "express": ["app.get", "app.post", "app.use", "router.", "req, res"],
            "node": ["require(", "module.exports", "process.", "Buffer"],
            "jest": ["describe(", "it(", "test(", "expect(", "beforeEach", "afterEach"],
            "mocha": ["describe(", "it(", "before(", "after("],
            "vue": ["Vue.", "computed:", "methods:", "mounted(", "created("],
            "angular": ["@Component", "@Injectable", "@NgModule"],
        }
