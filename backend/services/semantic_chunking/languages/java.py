"""Java language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_java as tsjava

from backend.services.semantic_chunking.languages.base import LanguageConfig


class JavaConfig(LanguageConfig):
    """Configuration for Java semantic chunking."""

    @property
    def language_name(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    @property
    def tree_sitter_language(self) -> Any:
        return tsjava.language()

    @property
    def function_query(self) -> str:
        # In Java, standalone functions don't exist - they're all methods
        # This is for methods outside of a specific class context search
        return """
        (method_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
            body: (block) @body
        ) @function
        """

    @property
    def class_query(self) -> str:
        return """
        [
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body
            ) @class

            (interface_declaration
                name: (identifier) @name
                body: (interface_body) @body
            ) @class

            (enum_declaration
                name: (identifier) @name
                body: (enum_body) @body
            ) @class
        ]
        """

    @property
    def method_query(self) -> str:
        return """
        (class_body
            (method_declaration
                name: (identifier) @method_name
                parameters: (formal_parameters) @params
                body: (block) @body
            ) @method
        )
        """

    @property
    def import_query(self) -> str:
        return """
        (import_declaration) @import
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract Javadoc comment."""
        current = node.prev_sibling
        while current:
            if current.type == "block_comment":
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
            elif current.type not in ("modifiers", "marker_annotation", "annotation"):
                break
            current = current.prev_sibling
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract method signature."""
        modifiers = []
        return_type = None
        name = None
        params = None

        for child in node.children:
            if child.type == "modifiers":
                modifiers.append(self.get_node_text(child, source_bytes))
            elif child.type in ("void_type", "type_identifier", "generic_type", "array_type"):
                return_type = self.get_node_text(child, source_bytes)
            elif child.type == "identifier":
                name = self.get_node_text(child, source_bytes)
            elif child.type == "formal_parameters":
                params = self.get_node_text(child, source_bytes)

        if name and params:
            parts = []
            if modifiers:
                parts.append(" ".join(modifiers))
            if return_type:
                parts.append(return_type)
            parts.append(f"{name}{params}")
            return " ".join(parts)
        return None

    def extract_decorators(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract annotations from a Java method/class."""
        annotations = []
        for child in node.children:
            if child.type == "modifiers":
                for mod_child in child.children:
                    if mod_child.type in ("marker_annotation", "annotation"):
                        annotations.append(self.get_node_text(mod_child, source_bytes))
        return annotations

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names."""
        params = []
        for child in node.children:
            if child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "formal_parameter":
                        for sub in param.children:
                            if sub.type == "identifier":
                                params.append(self.get_node_text(sub, source_bytes))
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type."""
        for child in node.children:
            if child.type in (
                "void_type",
                "type_identifier",
                "generic_type",
                "array_type",
                "integral_type",
                "floating_point_type",
                "boolean_type",
            ):
                return self.get_node_text(child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for Java behavior patterns."""
        return {
            "conditionals": "[( if_statement) (switch_expression) (ternary_expression)] @cond",
            "loops": "[( for_statement) (enhanced_for_statement) (while_statement) (do_statement)] @loop",
            "error_handling": "(try_statement) @try",
            "throws": "(throws) @throws",
            "synchronized": "(synchronized_statement) @sync",
            "lambda": "(lambda_expression) @lambda",
            "streams": "(method_invocation name: (identifier) @name (#match? @name \"^(stream|filter|map|reduce|collect)$\")) @stream",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting Java framework usage."""
        return {
            "spring": [
                "@Controller",
                "@Service",
                "@Repository",
                "@Autowired",
                "@RequestMapping",
                "@GetMapping",
                "@PostMapping",
                "@Bean",
                "@Configuration",
            ],
            "junit": ["@Test", "@BeforeEach", "@AfterEach", "@BeforeAll", "assertEquals"],
            "lombok": ["@Data", "@Getter", "@Setter", "@Builder", "@AllArgsConstructor"],
            "jpa": ["@Entity", "@Table", "@Column", "@Id", "@ManyToOne", "@OneToMany"],
            "jackson": ["@JsonProperty", "@JsonIgnore", "ObjectMapper"],
        }
