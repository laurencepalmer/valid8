"""Tests for the behavior analyzer module."""

import pytest
from backend.services.semantic_chunking.behavior_analyzer import BehaviorAnalyzer
from backend.services.semantic_chunking.ast_parser import ASTParser


@pytest.fixture
def analyzer():
    """Create a BehaviorAnalyzer instance."""
    return BehaviorAnalyzer()


@pytest.fixture
def parser():
    """Create an ASTParser instance."""
    return ASTParser()


class TestConditionalDetection:
    """Tests for conditional detection."""

    def test_detect_if_statement(self, analyzer, parser):
        """Test detecting if statements."""
        source = """
def check_value(x):
    if x > 0:
        return True
    return False
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_conditionals is True

    def test_no_conditionals(self, analyzer, parser):
        """Test function without conditionals."""
        source = """
def get_value():
    return 42
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_conditionals is False


class TestLoopDetection:
    """Tests for loop detection."""

    def test_detect_for_loop(self, analyzer, parser):
        """Test detecting for loops."""
        source = """
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_loops is True

    def test_detect_while_loop(self, analyzer, parser):
        """Test detecting while loops."""
        source = """
def countdown(n):
    while n > 0:
        n -= 1
    return n
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_loops is True

    def test_no_loops(self, analyzer, parser):
        """Test function without loops."""
        source = """
def get_value():
    return 42
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_loops is False


class TestErrorHandlingDetection:
    """Tests for error handling detection."""

    def test_detect_try_except(self, analyzer, parser):
        """Test detecting try/except blocks."""
        source = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_error_handling is True

    def test_no_error_handling(self, analyzer, parser):
        """Test function without error handling."""
        source = """
def divide(a, b):
    return a / b
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_error_handling is False


class TestAsyncDetection:
    """Tests for async detection."""

    def test_detect_async_function(self, analyzer, parser):
        """Test detecting async functions."""
        source = """
async def fetch_data():
    await some_api()
    return data
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_async is True

    def test_no_async(self, analyzer, parser):
        """Test sync function."""
        source = """
def get_data():
    return data
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.has_async is False


class TestReturnBehavior:
    """Tests for return behavior analysis."""

    def test_void_return(self, analyzer, parser):
        """Test function with no return."""
        source = """
def log_message(msg):
    print(msg)
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.return_behavior == "void"

    def test_value_return(self, analyzer, parser):
        """Test function with unconditional return."""
        source = """
def get_value():
    return 42
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.return_behavior == "value"

    def test_conditional_return(self, analyzer, parser):
        """Test function with conditional return."""
        source = """
def maybe_get(x):
    if x > 0:
        return x
    return None
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.return_behavior == "conditional"


class TestComplexityTier:
    """Tests for complexity tier calculation."""

    def test_simple_function(self, analyzer, parser):
        """Test simple function complexity."""
        source = """
def get_value():
    return 42
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.complexity_tier == "simple"

    def test_moderate_function(self, analyzer, parser):
        """Test moderate complexity function."""
        source = """
def process_items(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
    return result
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.complexity_tier in ("moderate", "simple")

    def test_complex_function(self, analyzer, parser):
        """Test complex function."""
        source = """
def complex_process(items):
    result = []
    for item in items:
        if item > 0:
            for sub in item.children:
                if sub.valid:
                    try:
                        result.append(sub.process())
                    except Exception:
                        pass
                elif sub.fallback:
                    result.append(sub.fallback())
    return result
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.complexity_tier == "complex"


class TestPurityDetection:
    """Tests for pure function detection."""

    def test_pure_function(self, analyzer, parser):
        """Test detecting pure function."""
        source = """
def add(a, b):
    return a + b
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        assert behavior.is_pure is True

    def test_impure_function_with_mutation(self, analyzer, parser):
        """Test detecting impure function with mutation via assignment."""
        source = """
def update_user(user, name):
    user.name = name
    return user
"""
        tree = parser.parse(source, "python")
        functions = parser.find_functions(tree, "python")
        assert len(functions) == 1

        behavior = analyzer.analyze(functions[0], source.encode(), "python")
        # Function assigns to a member - should detect mutation
        assert behavior.has_mutations is True
