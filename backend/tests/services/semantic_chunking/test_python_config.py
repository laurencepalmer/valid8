"""Tests for Python language configuration."""

import pytest
from backend.services.semantic_chunking.languages.python import PythonConfig
from backend.services.semantic_chunking.ast_parser import ASTParser


@pytest.fixture
def config():
    """Create a PythonConfig instance."""
    return PythonConfig()


@pytest.fixture
def parser():
    """Create an ASTParser instance."""
    return ASTParser()


class TestPythonConfig:
    """Tests for PythonConfig."""

    def test_language_name(self, config):
        """Test language name."""
        assert config.language_name == "python"

    def test_file_extensions(self, config):
        """Test file extensions."""
        assert ".py" in config.file_extensions
        assert ".pyi" in config.file_extensions

    def test_tree_sitter_language(self, config):
        """Test tree-sitter language is available."""
        assert config.tree_sitter_language is not None

    def test_is_tree_sitter_supported(self, config):
        """Test tree-sitter support flag."""
        assert config.is_tree_sitter_supported is True


class TestPythonDocstringExtraction:
    """Tests for Python docstring extraction."""

    def test_extract_triple_quote_docstring(self, config, parser):
        """Test extracting triple-quoted docstring."""
        source = '''
def hello():
    """This is a docstring."""
    pass
'''
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        docstring = config.extract_docstring(functions[0], source.encode())
        assert docstring == "This is a docstring."

    def test_extract_single_quote_docstring(self, config, parser):
        """Test extracting single-quoted docstring."""
        source = """
def hello():
    'This is a docstring.'
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        docstring = config.extract_docstring(functions[0], source.encode())
        assert docstring == "This is a docstring."

    def test_no_docstring(self, config, parser):
        """Test function without docstring."""
        source = """
def hello():
    x = 1
    return x
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        docstring = config.extract_docstring(functions[0], source.encode())
        assert docstring is None


class TestPythonSignatureExtraction:
    """Tests for Python signature extraction."""

    def test_extract_simple_signature(self, config, parser):
        """Test extracting simple function signature."""
        source = """
def hello():
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        signature = config.extract_signature(functions[0], source.encode())
        assert signature == "def hello()"

    def test_extract_signature_with_params(self, config, parser):
        """Test extracting signature with parameters."""
        source = """
def greet(name, greeting="Hello"):
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        signature = config.extract_signature(functions[0], source.encode())
        assert "def greet(" in signature
        assert "name" in signature

    def test_extract_signature_with_return_type(self, config, parser):
        """Test extracting signature with return type."""
        source = """
def get_value() -> int:
    return 42
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        signature = config.extract_signature(functions[0], source.encode())
        assert "-> int" in signature


class TestPythonParameterExtraction:
    """Tests for Python parameter extraction."""

    def test_extract_simple_params(self, config, parser):
        """Test extracting simple parameters."""
        source = """
def greet(name, message):
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        params = config.extract_parameters(functions[0], source.encode())
        assert "name" in params
        assert "message" in params

    def test_exclude_self(self, config, parser):
        """Test that self is excluded from parameters."""
        source = """
class Greeter:
    def greet(self, name):
        pass
"""
        tree = parser.parse(source, "python")
        # Find the method directly
        methods = parser.find_methods(tree, "python")
        assert len(methods) == 1

        params = config.extract_parameters(methods[0], source.encode())
        assert "self" not in params
        assert "name" in params

    def test_typed_params(self, config, parser):
        """Test extracting typed parameters."""
        source = """
def greet(name: str, count: int = 1):
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        params = config.extract_parameters(functions[0], source.encode())
        assert "name" in params
        assert "count" in params


class TestPythonDecoratorExtraction:
    """Tests for Python decorator extraction."""

    def test_extract_decorators(self, config, parser):
        """Test extracting decorators."""
        source = """
@staticmethod
@property
def get_value():
    pass
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        decorators = config.extract_decorators(functions[0], source.encode())
        assert len(decorators) == 2
        assert "@staticmethod" in decorators
        assert "@property" in decorators


class TestPythonFrameworkPatterns:
    """Tests for Python framework pattern detection."""

    def test_fastapi_patterns(self, config):
        """Test FastAPI pattern detection."""
        patterns = config.get_framework_patterns()
        assert "fastapi" in patterns
        assert "@app.get" in patterns["fastapi"]
        assert "@router." in patterns["fastapi"]

    def test_pytest_patterns(self, config):
        """Test pytest pattern detection."""
        patterns = config.get_framework_patterns()
        assert "pytest" in patterns
        assert "@pytest." in patterns["pytest"]
        assert "def test_" in patterns["pytest"]

    def test_pydantic_patterns(self, config):
        """Test Pydantic pattern detection."""
        patterns = config.get_framework_patterns()
        assert "pydantic" in patterns
        assert "BaseModel" in patterns["pydantic"]


class TestPythonClassExtraction:
    """Tests for Python class extraction."""

    def test_find_classes(self, config, parser):
        """Test finding class definitions."""
        source = """
class MyClass:
    pass

class AnotherClass(BaseClass):
    def method(self):
        pass
"""
        tree = parser.parse(source, "python")
        classes = parser.find_classes(tree, "python")
        assert len(classes) == 2

    def test_extract_class_docstring(self, config, parser):
        """Test extracting class docstring."""
        source = '''
class MyClass:
    """This is a class docstring."""
    pass
'''
        tree = parser.parse(source, "python")
        classes = parser.find_classes(tree, "python")
        assert len(classes) == 1

        docstring = config.extract_docstring(classes[0], source.encode())
        assert docstring == "This is a class docstring."
