"""Go language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_go as tsgo

from backend.services.semantic_chunking.languages.base import LanguageConfig


class GoConfig(LanguageConfig):
    """Configuration for Go semantic chunking."""

    @property
    def language_name(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> list[str]:
        return [".go"]

    @property
    def tree_sitter_language(self) -> Any:
        return tsgo.language()

    @property
    def function_query(self) -> str:
        return """
        (function_declaration
            name: (identifier) @name
            parameters: (parameter_list) @params
            result: (_)? @return_type
            body: (block) @body
        ) @function
        """

    @property
    def class_query(self) -> str:
        # Go doesn't have classes, but has type declarations with methods
        return """
        (type_declaration
            (type_spec
                name: (type_identifier) @name
                type: (struct_type) @body
            )
        ) @class
        """

    @property
    def method_query(self) -> str:
        return """
        (method_declaration
            receiver: (parameter_list) @receiver
            name: (field_identifier) @method_name
            parameters: (parameter_list) @params
            result: (_)? @return_type
            body: (block) @body
        ) @method
        """

    @property
    def import_query(self) -> str:
        return """
        (import_declaration) @import
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract Go doc comment."""
        current = node.prev_sibling
        comments = []
        while current:
            if current.type == "comment":
                comment = self.get_node_text(current, source_bytes)
                # Remove // prefix
                if comment.startswith("//"):
                    comment = comment[2:].strip()
                comments.insert(0, comment)
            else:
                break
            current = current.prev_sibling

        if comments:
            return " ".join(comments)
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function/method signature."""
        name = None
        params = None
        return_type = None
        receiver = None

        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_text(child, source_bytes)
            elif child.type == "field_identifier":
                name = self.get_node_text(child, source_bytes)
            elif child.type == "parameter_list":
                if receiver is None and node.type == "method_declaration":
                    receiver = self.get_node_text(child, source_bytes)
                else:
                    params = self.get_node_text(child, source_bytes)
            elif child.type in (
                "type_identifier",
                "pointer_type",
                "slice_type",
                "map_type",
                "parameter_list",
            ):
                if params is not None:  # This is the return type
                    return_type = self.get_node_text(child, source_bytes)

        if name and params:
            if receiver:
                sig = f"func {receiver} {name}{params}"
            else:
                sig = f"func {name}{params}"
            if return_type:
                sig += f" {return_type}"
            return sig
        return None

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names."""
        params = []
        found_receiver = False

        for child in node.children:
            if child.type == "parameter_list":
                if not found_receiver and node.type == "method_declaration":
                    found_receiver = True
                    continue
                for param in child.children:
                    if param.type == "parameter_declaration":
                        for sub in param.children:
                            if sub.type == "identifier":
                                params.append(self.get_node_text(sub, source_bytes))
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type."""
        # Find the result node
        for i, child in enumerate(node.children):
            if child.type == "parameter_list":
                # Look for return type after parameters
                for j in range(i + 1, len(node.children)):
                    next_child = node.children[j]
                    if next_child.type != "block":
                        return self.get_node_text(next_child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for Go behavior patterns."""
        return {
            "conditionals": "[( if_statement) (expression_switch_statement) (type_switch_statement)] @cond",
            "loops": "[( for_statement) (range_clause)] @loop",
            "error_handling": "(if_statement consequence: (block (return_statement))) @error_check",
            "defer": "(defer_statement) @defer",
            "goroutine": "(go_statement) @goroutine",
            "channel": "[( send_statement) (receive_expression)] @channel",
            "select": "(select_statement) @select",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting Go framework usage."""
        return {
            "gin": ["gin.Context", "gin.Engine", "c.JSON", "c.Bind"],
            "echo": ["echo.Context", "echo.Echo", "c.JSON"],
            "fiber": ["fiber.Ctx", "fiber.App"],
            "gorm": ["gorm.DB", "gorm.Model", "db.Create", "db.Find"],
            "testing": ["testing.T", "t.Run", "t.Error", "t.Fatal"],
            "context": ["context.Context", "ctx.Done", "ctx.Err"],
        }
