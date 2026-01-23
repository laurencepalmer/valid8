"""Rust language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_rust as tsrust

from backend.services.semantic_chunking.languages.base import LanguageConfig


class RustConfig(LanguageConfig):
    """Configuration for Rust semantic chunking."""

    @property
    def language_name(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> list[str]:
        return [".rs"]

    @property
    def tree_sitter_language(self) -> Any:
        return tsrust.language()

    @property
    def function_query(self) -> str:
        return """
        (function_item
            name: (identifier) @name
            parameters: (parameters) @params
            return_type: (type_identifier)? @return_type
            body: (block) @body
        ) @function
        """

    @property
    def class_query(self) -> str:
        return """
        [
            (struct_item
                name: (type_identifier) @name
                body: (field_declaration_list)? @body
            ) @class

            (enum_item
                name: (type_identifier) @name
                body: (enum_variant_list) @body
            ) @class

            (trait_item
                name: (type_identifier) @name
                body: (declaration_list) @body
            ) @class
        ]
        """

    @property
    def method_query(self) -> str:
        return """
        (impl_item
            (declaration_list
                (function_item
                    name: (identifier) @method_name
                    parameters: (parameters) @params
                    return_type: (_)? @return_type
                    body: (block) @body
                ) @method
            )
        )
        """

    @property
    def import_query(self) -> str:
        return """
        (use_declaration) @import
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract doc comment (/// or //!)."""
        current = node.prev_sibling
        comments = []
        while current:
            if current.type == "line_comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("///") or comment.startswith("//!"):
                    comment = comment[3:].strip()
                    comments.insert(0, comment)
                else:
                    break
            elif current.type == "block_comment":
                comment = self.get_node_text(current, source_bytes)
                if comment.startswith("/**") or comment.startswith("/*!"):
                    comment = comment[3:-2].strip()
                    comments.insert(0, comment)
                else:
                    break
            else:
                break
            current = current.prev_sibling

        if comments:
            return " ".join(comments)
        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function signature."""
        visibility = None
        async_kw = False
        name = None
        params = None
        return_type = None

        for child in node.children:
            if child.type == "visibility_modifier":
                visibility = self.get_node_text(child, source_bytes)
            elif child.type == "async":
                async_kw = True
            elif child.type == "identifier":
                name = self.get_node_text(child, source_bytes)
            elif child.type == "parameters":
                params = self.get_node_text(child, source_bytes)
            elif child.type in ("type_identifier", "generic_type", "reference_type"):
                return_type = self.get_node_text(child, source_bytes)

        if name and params:
            parts = []
            if visibility:
                parts.append(visibility)
            if async_kw:
                parts.append("async")
            parts.append("fn")
            parts.append(f"{name}{params}")
            if return_type:
                parts.append(f"-> {return_type}")
            return " ".join(parts)
        return None

    def extract_decorators(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract attributes from a Rust function/struct."""
        attributes = []
        current = node.prev_sibling
        while current:
            if current.type == "attribute_item":
                attributes.insert(0, self.get_node_text(current, source_bytes))
            else:
                break
            current = current.prev_sibling
        return attributes

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names."""
        params = []
        for child in node.children:
            if child.type == "parameters":
                for param in child.children:
                    if param.type == "parameter":
                        for sub in param.children:
                            if sub.type == "identifier":
                                param_name = self.get_node_text(sub, source_bytes)
                                if param_name not in ("self", "&self", "&mut self"):
                                    params.append(param_name)
                    elif param.type == "self_parameter":
                        pass  # Skip self
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type."""
        for child in node.children:
            if child.type in (
                "type_identifier",
                "generic_type",
                "reference_type",
                "tuple_type",
                "unit_type",
            ):
                return self.get_node_text(child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for Rust behavior patterns."""
        return {
            "conditionals": "[( if_expression) (match_expression)] @cond",
            "loops": "[( for_expression) (while_expression) (loop_expression)] @loop",
            "error_handling": "(try_expression) @try",
            "async": "[( async) (await_expression)] @async",
            "unsafe": "(unsafe_block) @unsafe",
            "lifetime": "(lifetime) @lifetime",
            "pattern_matching": "(match_expression) @match",
            "option_result": "(call_expression function: (field_expression field: (field_identifier) @field (#match? @field \"^(unwrap|expect|ok|err|map|and_then)$\"))) @opt_result",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting Rust framework usage."""
        return {
            "tokio": ["#[tokio::main]", "tokio::spawn", "tokio::select"],
            "actix": ["#[actix_web", "HttpResponse", "web::"],
            "rocket": ["#[get", "#[post", "rocket::"],
            "axum": ["axum::Router", "axum::extract"],
            "serde": ["#[derive(Serialize", "#[derive(Deserialize", "serde_json"],
            "clap": ["#[derive(Parser", "clap::"],
        }
