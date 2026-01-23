"""Tests for the text generator module."""

import pytest
from backend.services.semantic_chunking.text_generator import TextGenerator
from backend.models.semantic_chunk import (
    SymbolType,
    StructuralMetadata,
    BehaviorDescriptors,
    DependencyInfo,
)


@pytest.fixture
def generator():
    """Create a TextGenerator instance."""
    return TextGenerator()


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return StructuralMetadata(
        symbol_type=SymbolType.FUNCTION,
        symbol_name="get_user_by_id",
        qualified_name="get_user_by_id",
        file_path="/path/to/user.py",
        relative_path="services/user.py",
        language="python",
        start_line=10,
        end_line=25,
        parameters=["user_id"],
        return_type="Optional[User]",
    )


@pytest.fixture
def sample_behavior():
    """Create sample behavior descriptors."""
    return BehaviorDescriptors(
        has_conditionals=True,
        has_loops=False,
        has_error_handling=True,
        has_async=False,
        complexity_tier="moderate",
    )


@pytest.fixture
def sample_dependencies():
    """Create sample dependencies."""
    return DependencyInfo(
        imports=["typing.Optional"],
        external_calls=["db.query"],
        framework_patterns=["sqlalchemy"],
    )


class TestGenerateSearchText:
    """Tests for search text generation."""

    def test_basic_search_text(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test basic search text generation."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
        )

        # Check that key elements are present
        assert "function" in search_text.lower()
        assert "get user by id" in search_text.lower()
        assert "python" in search_text.lower()
        assert "services/user.py" in search_text

    def test_search_text_includes_behavior(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that search text includes behavior descriptions."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
        )

        assert "conditional" in search_text.lower()
        assert "error" in search_text.lower()

    def test_search_text_includes_parameters(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that search text includes parameter information."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
        )

        assert "parameter" in search_text.lower()
        assert "user_id" in search_text

    def test_search_text_includes_return_type(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that search text includes return type."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
        )

        assert "optional" in search_text.lower() or "user" in search_text.lower()

    def test_search_text_includes_framework(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that search text includes framework patterns."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
        )

        assert "sqlalchemy" in search_text.lower()

    def test_search_text_includes_docstring(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that search text includes docstring."""
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
            docstring="Fetch user from database by their unique ID.",
        )

        assert "fetch user from database" in search_text.lower()

    def test_search_text_truncates_long_docstring(self, generator, sample_metadata, sample_behavior, sample_dependencies):
        """Test that long docstrings are truncated."""
        long_docstring = "A" * 500
        search_text = generator.generate_search_text(
            metadata=sample_metadata,
            behavior=sample_behavior,
            dependencies=sample_dependencies,
            docstring=long_docstring,
        )

        assert "..." in search_text
        assert len(search_text) < 1000


class TestDescribeSymbol:
    """Tests for symbol description generation."""

    def test_function_description(self, generator, sample_metadata):
        """Test function symbol description."""
        description = generator._describe_symbol(sample_metadata)
        assert "function" in description.lower()
        assert "python" in description.lower()

    def test_method_description(self, generator):
        """Test method symbol description with parent scope."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.METHOD,
            symbol_name="get_name",
            qualified_name="User.get_name",
            file_path="/path/to/user.py",
            relative_path="models/user.py",
            language="python",
            start_line=20,
            end_line=25,
            parent_scope="User",
        )

        description = generator._describe_symbol(metadata)
        assert "method" in description.lower()
        assert "user" in description.lower()

    def test_class_description(self, generator):
        """Test class symbol description."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.CLASS,
            symbol_name="UserService",
            qualified_name="UserService",
            file_path="/path/to/user.py",
            relative_path="services/user.py",
            language="python",
            start_line=1,
            end_line=100,
        )

        description = generator._describe_symbol(metadata)
        assert "class" in description.lower()
        assert "user service" in description.lower()


class TestDescribeSignature:
    """Tests for signature description generation."""

    def test_no_parameters(self, generator):
        """Test signature with no parameters."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.FUNCTION,
            symbol_name="get_version",
            qualified_name="get_version",
            file_path="/path/to/utils.py",
            relative_path="utils.py",
            language="python",
            start_line=1,
            end_line=3,
            parameters=[],
        )

        signature_desc = generator._describe_signature(metadata)
        assert signature_desc is not None
        assert "no parameter" in signature_desc.lower()

    def test_single_parameter(self, generator):
        """Test signature with single parameter."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.FUNCTION,
            symbol_name="get_user",
            qualified_name="get_user",
            file_path="/path/to/user.py",
            relative_path="user.py",
            language="python",
            start_line=1,
            end_line=5,
            parameters=["user_id"],
        )

        signature_desc = generator._describe_signature(metadata)
        assert signature_desc is not None
        assert "1 parameter" in signature_desc
        assert "user_id" in signature_desc

    def test_multiple_parameters(self, generator):
        """Test signature with multiple parameters."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.FUNCTION,
            symbol_name="create_user",
            qualified_name="create_user",
            file_path="/path/to/user.py",
            relative_path="user.py",
            language="python",
            start_line=1,
            end_line=10,
            parameters=["name", "email", "password"],
        )

        signature_desc = generator._describe_signature(metadata)
        assert signature_desc is not None
        assert "3 parameters" in signature_desc

    def test_class_has_no_signature(self, generator):
        """Test that classes don't have signature descriptions."""
        metadata = StructuralMetadata(
            symbol_type=SymbolType.CLASS,
            symbol_name="User",
            qualified_name="User",
            file_path="/path/to/user.py",
            relative_path="user.py",
            language="python",
            start_line=1,
            end_line=50,
        )

        signature_desc = generator._describe_signature(metadata)
        assert signature_desc is None


class TestDescribeBehavior:
    """Tests for behavior description generation."""

    def test_no_behavior(self, generator):
        """Test function with no special behavior."""
        behavior = BehaviorDescriptors()
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is None

    def test_conditional_behavior(self, generator):
        """Test function with conditionals."""
        behavior = BehaviorDescriptors(has_conditionals=True)
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is not None
        assert "conditional" in behavior_desc.lower()

    def test_async_behavior(self, generator):
        """Test async function."""
        behavior = BehaviorDescriptors(has_async=True)
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is not None
        assert "async" in behavior_desc.lower()

    def test_error_handling_behavior(self, generator):
        """Test function with error handling."""
        behavior = BehaviorDescriptors(has_error_handling=True)
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is not None
        assert "error" in behavior_desc.lower()

    def test_pure_function_behavior(self, generator):
        """Test pure function."""
        behavior = BehaviorDescriptors(is_pure=True)
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is not None
        assert "pure" in behavior_desc.lower()

    def test_multiple_behaviors(self, generator):
        """Test function with multiple behaviors."""
        behavior = BehaviorDescriptors(
            has_conditionals=True,
            has_loops=True,
            has_error_handling=True,
        )
        behavior_desc = generator._describe_behavior(behavior)
        assert behavior_desc is not None
        assert "conditional" in behavior_desc.lower()
        assert "loop" in behavior_desc.lower()
        assert "error" in behavior_desc.lower()


class TestGenerateChunkId:
    """Tests for chunk ID generation."""

    def test_basic_chunk_id(self, generator, sample_metadata):
        """Test basic chunk ID generation."""
        chunk_id = generator.generate_chunk_id(sample_metadata)
        assert "services/user.py" in chunk_id
        assert "get_user_by_id" in chunk_id
        assert "10" in chunk_id

    def test_sub_chunk_id(self, generator, sample_metadata):
        """Test sub-chunk ID generation."""
        chunk_id = generator.generate_chunk_id(sample_metadata, sub_chunk_index=2)
        assert "-2" in chunk_id

    def test_chunk_id_format(self, generator, sample_metadata):
        """Test chunk ID format."""
        chunk_id = generator.generate_chunk_id(sample_metadata)
        # Format should be: {relative_path}:{qualified_name}:{start_line}
        parts = chunk_id.split(":")
        assert len(parts) == 3
        assert parts[0] == "services/user.py"
        assert parts[1] == "get_user_by_id"
        assert parts[2] == "10"
