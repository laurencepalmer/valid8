"""Python language configuration for semantic chunking."""

from typing import Any, Optional

import tree_sitter_python as tspython

from backend.services.semantic_chunking.languages.base import LanguageConfig


class PythonConfig(LanguageConfig):
    """Configuration for Python semantic chunking."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py", ".pyi"]

    @property
    def tree_sitter_language(self) -> Any:
        return tspython.language()

    @property
    def function_query(self) -> str:
        return """
        (function_definition
            name: (identifier) @name
            parameters: (parameters) @params
            return_type: (type)? @return_type
            body: (block) @body
        ) @function
        """

    @property
    def class_query(self) -> str:
        return """
        (class_definition
            name: (identifier) @name
            superclasses: (argument_list)? @bases
            body: (block) @body
        ) @class
        """

    @property
    def method_query(self) -> str:
        return """
        (class_definition
            body: (block
                (function_definition
                    name: (identifier) @method_name
                    parameters: (parameters) @params
                    return_type: (type)? @return_type
                    body: (block) @body
                ) @method
            )
        )
        """

    @property
    def import_query(self) -> str:
        return """
        [
            (import_statement) @import
            (import_from_statement) @import
        ]
        """

    def extract_docstring(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract docstring from Python function/class."""
        # Look for the body block
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break

        if not body or len(body.children) == 0:
            return None

        # First statement in body - check if it's a string
        first_stmt = None
        for child in body.children:
            if child.type == "expression_statement":
                first_stmt = child
                break

        if not first_stmt:
            return None

        # Check if it's a string literal
        for child in first_stmt.children:
            if child.type == "string":
                docstring = self.get_node_text(child, source_bytes)
                # Clean up the docstring (remove quotes)
                if docstring.startswith('"""') or docstring.startswith("'''"):
                    docstring = docstring[3:-3]
                elif docstring.startswith('"') or docstring.startswith("'"):
                    docstring = docstring[1:-1]
                return docstring.strip()

        return None

    def extract_signature(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract function/method signature."""
        name = None
        params = None
        return_type = None

        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_text(child, source_bytes)
            elif child.type == "parameters":
                params = self.get_node_text(child, source_bytes)
            elif child.type == "type":
                return_type = self.get_node_text(child, source_bytes)

        if name and params:
            sig = f"def {name}{params}"
            if return_type:
                sig += f" -> {return_type}"
            return sig
        return None

    def extract_decorators(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract decorators from a Python function/class."""
        decorators = []
        # Check previous siblings for decorators
        current = node.prev_sibling
        while current:
            if current.type == "decorator":
                dec_text = self.get_node_text(current, source_bytes)
                decorators.insert(0, dec_text)
            elif current.type != "comment":
                break
            current = current.prev_sibling
        return decorators

    def extract_parameters(self, node: Any, source_bytes: bytes) -> list[str]:
        """Extract parameter names from a function/method."""
        params = []
        for child in node.children:
            if child.type == "parameters":
                for param_child in child.children:
                    if param_child.type == "identifier":
                        param_name = self.get_node_text(param_child, source_bytes)
                        if param_name != "self" and param_name != "cls":
                            params.append(param_name)
                    elif param_child.type in (
                        "typed_parameter",
                        "default_parameter",
                        "typed_default_parameter",
                    ):
                        # Get the identifier from within the typed parameter
                        for sub in param_child.children:
                            if sub.type == "identifier":
                                param_name = self.get_node_text(sub, source_bytes)
                                if param_name != "self" and param_name != "cls":
                                    params.append(param_name)
                                break
        return params

    def extract_return_type(self, node: Any, source_bytes: bytes) -> Optional[str]:
        """Extract return type annotation."""
        for child in node.children:
            if child.type == "type":
                return self.get_node_text(child, source_bytes)
        return None

    def get_behavior_patterns(self) -> dict[str, str]:
        """Return tree-sitter queries for Python behavior patterns."""
        return {
            "conditionals": "(if_statement) @if",
            "loops": "[( for_statement) (while_statement)] @loop",
            "error_handling": "(try_statement) @try",
            "async": "(function_definition (async)) @async",
            "assertions": "(assert_statement) @assert",
            "raise": "(raise_statement) @raise",
            "yield": "(yield) @yield",
            "with": "(with_statement) @with",
            "list_comprehension": "(list_comprehension) @listcomp",
            "generator": "(generator_expression) @genexp",
        }

    def get_framework_patterns(self) -> dict[str, list[str]]:
        """Return patterns for detecting Python framework usage."""
        return {
            "fastapi": ["@app.get", "@app.post", "@router.", "FastAPI", "APIRouter"],
            "flask": ["@app.route", "Flask", "@blueprint."],
            "django": ["@login_required", "HttpResponse", "render", "models.Model"],
            "pytest": ["@pytest.", "def test_", "@fixture"],
            "sqlalchemy": ["Base", "Column", "relationship", "Session"],
            "pydantic": ["BaseModel", "Field", "validator"],
            "asyncio": ["async def", "await", "asyncio."],
            "dataclasses": ["@dataclass"],
            "typing": ["Optional", "Union", "List", "Dict", "Callable"],
        }
