"""TypeScript language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_typescript as tstypescript

from backend.services.semantic_chunking.languages.base import LanguageConfig


class TypeScriptConfig(LanguageConfig):
    """Configuration for TypeScript semantic chunking."""

    @property
    def language_name(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".mts", ".cts"]

    @property
    def tree_sitter_language(self) -> Any:
        return tstypescript.language_typescript()

    @property
    def function_query(self) -> str:
        return """
        [
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                return_type: (type_annotation)? @return_type
                body: (statement_block) @body
            ) @function

            (variable_declarator
                name: (identifier) @name
                value: (arrow_function
                    parameters: (formal_parameters) @params
                    return_type: (type_annotation)? @return_type
                    body: (_) @body
                )
            ) @function

            (lexical_declaration
                (variable_declarator
                    name: (identifier) @name
                    value: (arrow_function
                        parameters: (formal_parameters) @params
                        return_type: (type_annotation)? @return_type
                        body: (_) @body
                    )
                ) @function
            )

            (export_statement
                (function_declaration
                    name: (identifier) @name
                    parameters: (formal_parameters) @params
                    return_type: (type_annotation)? @return_type
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
                name: (type_identifier) @name
                body: (class_body) @body
            ) @class

            (export_statement
                (class_declaration
                    name: (type_identifier) @name
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
                return_type: (type_annotation)? @return_type
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
        """Extract JSDoc/TSDoc comment."""
        current = node.prev_sibling
        while current:
            if current.type == "comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("/**"):
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
            elif current.type not in ("export_statement", "decorator"):
                break
            current = current.prev_sibling
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function signature with types."""
        if node.type == "function_declaration":
            name = None
            params = None
            return_type = None
            for child in node.children:
                if child.type == "identifier":
                    name = self.get_node_text(child, source_bytes)
                elif child.type == "formal_parameters":
                    params = self.get_node_text(child, source_bytes)
                elif child.type == "type_annotation":
                    return_type = self.get_node_text(child, source_bytes)
            if name and params:
                sig = f"function {name}{params}"
                if return_type:
                    sig += return_type
                return sig
        return None

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type annotation."""
        for child in node.children:
            if child.type == "type_annotation":
                text = self.get_node_text(child, source_bytes)
                # Remove leading colon and whitespace
                return text.lstrip(": ").strip()
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
                        elif param.type == "required_parameter":
                            for sub in param.children:
                                if sub.type == "identifier":
                                    params.append(self.get_node_text(sub, source_bytes))
                                    break
                        elif param.type == "optional_parameter":
                            for sub in param.children:
                                if sub.type == "identifier":
                                    params.append(self.get_node_text(sub, source_bytes))
                                    break
                elif child.type in ("arrow_function", "function_expression"):
                    find_params(child)

        find_params(node)
        return params

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for TypeScript behavior patterns."""
        return {
            "conditionals": "[( if_statement) (ternary_expression)] @cond",
            "loops": "[( for_statement) (for_in_statement) (while_statement) (do_statement)] @loop",
            "error_handling": "(try_statement) @try",
            "async": "[( async) (await_expression)] @async",
            "type_guards": "(type_predicate) @guard",
            "generics": "(type_parameters) @generics",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting TypeScript framework usage."""
        return {
            "react": [
                "useState",
                "useEffect",
                "useContext",
                "React.FC",
                "React.Component",
            ],
            "angular": [
                "@Component",
                "@Injectable",
                "@NgModule",
                "@Input",
                "@Output",
            ],
            "nestjs": [
                "@Controller",
                "@Get",
                "@Post",
                "@Injectable",
                "@Module",
            ],
            "express": ["Request", "Response", "NextFunction", "Router"],
            "typeorm": ["@Entity", "@Column", "@Repository", "getRepository"],
            "prisma": ["PrismaClient", "prisma."],
        }
